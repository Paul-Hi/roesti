#include <app/application.hpp>
#include <app/parameter.hpp>
#include <core/image.hpp>
#include <core/types.hpp>
#include <imgui.h>
#include <imgui_internal.h>
#include <implot.h>
#include <implot3d.h>
#include <ui/imguiBackend.hpp>
#include <ui/log.hpp>

#include "kernels.cuh"
#include "roesti.hpp"

#include "cuda_measurable_event.cuh"

using namespace saf;
using namespace roesti;
using namespace roesti::utils;
using namespace torch;

class TestLayer : public Layer
{
public:
    virtual void onAttach(Application* application) override
    {
        const IVec2& res = mResolution;

        // Create vertex data for two quads (each quad = 2 triangles = 6 vertices)
        vertexPositions = torch::tensor({
                                            { -0.6f, 0.6f }, // Top-left
                                            { 0.0f, 0.6f },  // Top-right
                                            { 0.0f, 0.0f },  // Bottom-right
                                            { -0.6f, 0.0f }, // Bottom-left

                                            { 0.2f, 0.4f },  // Top-left
                                            { 0.8f, 0.4f },  // Top-right
                                            { 0.8f, -0.2f }, // Bottom-right
                                            { 0.2f, -0.2f }  // Bottom-left
                                        },
                                        torch::kFloat)
                              .cuda();

        // Assign colors to each quad (4 vertices per quad)
        vertexColors = torch::tensor({ // Quad 1: Red
                                       { 1.0f, 0.0f, 0.0f, 0.5f },
                                       { 1.0f, 0.0f, 0.0f, 0.5f },
                                       { 1.0f, 0.0f, 0.0f, 0.5f },
                                       { 1.0f, 0.0f, 0.0f, 0.5f },
                                       // Quad 2: Green
                                       { 0.0f, 1.0f, 0.0f, 0.5f },
                                       { 0.0f, 1.0f, 0.0f, 0.5f },
                                       { 0.0f, 1.0f, 0.0f, 0.5f },
                                       { 0.0f, 1.0f, 0.0f, 0.5f } },
                                     torch::kFloat)
                           .cuda();

        vertexTexcoords = torch::zeros({ 8, 2 }, torch::kFloat).cuda();

        // Assign layers: Quad 1 -> 0, Quad 2 -> 1
        vertexLayers = torch::tensor({ 0, 0, 0, 0, 1, 1, 1, 1 }, torch::kInt).cuda();

        // Index buffer for triangle strip to draw two quads (each quad = 2 triangles = 4 indices per quad)
        // Triangle strip order for two quads:
        // Quad 1: 0, 1, 3, 2
        // Degenerate: 2, 2, 4, 4
        // Quad 2: 4, 5, 7, 6
        indexBuffer = torch::tensor({
                                        0, 1, 3, 2, // Quad 1
                                        2, 4,
                                        4, 5, 7, 6, // Quad 2
                                    },
                                    torch::kInt)
                          .cuda();

        backgroundColor = torch::tensor({ 0, 0, 0, 1 }, torch::kFloat).cuda();

        mOutputRT = std::make_shared<Image>(application->getApplicationContext(), res.x(), res.y(), VK_FORMAT_R32G32B32A32_SFLOAT, nullptr, true);

        renderTarget = torch::zeros({ res.y(), res.x(), 4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        mRasterizer = std::make_shared<Rasterizer2D>();

        PipelineConfig config;
        config.width         = res.x();
        config.height        = res.y();
        config.blendMode     = BlendMode::AlphaBlendF2B;
        config.pointSize     = 12;
        config.lineWidth     = 2;
        config.wireframe     = false;
        config.outputRT      = renderTarget;
        config.primitiveType = PrimitiveType::TriangleStrip;

        mRasterizer->createPipeline(config, 0);

        mRasterizer->addVertexBuffer(vertexPositions, vertexColors, vertexTexcoords, vertexLayers, vertexPositions.size(0), 0);
        mRasterizer->addIndexBuffer(indexBuffer, indexBuffer.size(0), 0);
        mRasterizer->setClear(backgroundColor, 0);

        Statistics& stats = Statistics::get();
        stats.setMaximumSamplesPerEntry(60);
    }

    virtual void onDetach() override
    {
    }

    virtual void onUpdate(Application* application, F32 dt) override
    {
        (void)application;
        mOutputRT->awaitCudaUpdateClearance();

        IVec2& res = mResolution;

        // measure rasterization and blit time
        {
            CUDA_MEASURABLE_EVENT("Rasterization");
            mRasterizer->bindPipeline(0);
            mRasterizer->bindVertexBufferObject(0);
            mRasterizer->drawIndexed();
        }

        {
            CUDA_MEASURABLE_EVENT("Blit");
            copyToCudaSurface(renderTarget, mOutputRT->getCudaSurfaceObject(), res.x(), res.y());
            CUDA_CHECK_THROW(cudaGetLastError());
        }

        // Incremental rotation update
        auto deltaAngle     = torch::tensor(dt, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        auto cosA           = torch::cos(deltaAngle);
        auto sinA           = torch::sin(deltaAngle);
        auto rotationMatrix = torch::stack({ torch::stack({ cosA, -sinA }),
                                             torch::stack({ sinA, cosA }) })
                                  .to(torch::kFloat32)
                                  .cuda();

        // Centers for each quad
        auto quadC0 = vertexPositions.index({ torch::indexing::Slice(0, 4) }).mean(0, true);
        auto quadC1 = vertexPositions.index({ torch::indexing::Slice(4, 8) }).mean(0, true);

        // Rotate each quad around its own center
        vertexPositions.index_put_(
            { torch::indexing::Slice(0, 4) },
            torch::matmul(
                vertexPositions.index({ torch::indexing::Slice(0, 4) }) - quadC0,
                rotationMatrix.transpose(0, 1)) +
                quadC0);
        vertexPositions.index_put_(
            { torch::indexing::Slice(4, 8) },
            torch::matmul(
                vertexPositions.index({ torch::indexing::Slice(4, 8) }) - quadC1,
                rotationMatrix.transpose(0, 1)) +
                quadC1);

        mOutputRT->signalVulkanUpdateClearance();
    }

    virtual void onUIRender(Application* application) override
    {
        ImGui::Begin("Rösti");

        ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 4.0f);
        if (ImPlot::BeginPlot("##Image", ImVec2(-1, -1), ImPlotFlags_Equal))
        {
            const IVec2& res = mResolution;
            ImVec2 bmin(-res.x() * 0.5, -res.y() * 0.5);
            ImVec2 bmax(res.x() * 0.5, res.y() * 0.5);
            ImPlot::PlotImage("Output", reinterpret_cast<ImTextureID>(mOutputRT->getDescriptorSet()), bmin, bmax);

            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
        ImGui::End();

        Statistics& stats = Statistics::get();

        ImGui::Begin("Settings");
        auto rasterizationSample = stats.getSample("Rasterization");
        ImGui::Text("Rasterization Time (ms): %.3f", rasterizationSample ? rasterizationSample->mean : 0.0f);
        auto blitSample = stats.getSample("Blit");
        ImGui::Text("Blit Time (ms): %.3f", blitSample ? blitSample->mean : 0.0f);
        F64 fr = static_cast<F64>(ImGui::GetIO().Framerate);
        ImGui::Text("Approx. Frametime (ms): %.3f", 1000.0 / fr);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::End();

        UILog::get().render("Log");
    }

private:
    std::shared_ptr<Image> mOutputRT;

    Tensor renderTarget;

    IVec2 mResolution = IVec2(1920, 1080);

    std::shared_ptr<Rasterizer2D> mRasterizer;

    Tensor vertexPositions;
    Tensor vertexColors;
    Tensor vertexTexcoords;
    Tensor vertexLayers;

    Tensor indexBuffer;

    Tensor backgroundColor;
};

I32 main(I32 argc, char** argv)
{
    (void)argc;
    (void)argv;

    ApplicationSettings settings;
    settings.windowWidth  = 1920;
    settings.windowHeight = 1080;
    settings.fontSize     = 24.0f;
    settings.name         = "Rösti";
    settings.clearColor   = Vec4(0.3f, 0.3f, 0.3f, 1.0f);

    Application app(settings);

    app.setMenubarCallback([&app]()
                           {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Exit"))
            {
                app.close();
            }
            ImGui::EndMenu();
        } });

    app.pushLayer<TestLayer>();

    app.run();
}
