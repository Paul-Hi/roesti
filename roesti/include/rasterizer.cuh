#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <cuda_surface_types.h>
#include <cuda_texture_types.h>
#include <torch/torch.h>

namespace roesti
{
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT)

    enum class PrimitiveType : uint8_t
    {
        Points        = 0,
        Lines         = 1,
        LineStrip     = 2,
        Triangles     = 3,
        TriangleStrip = 4
    };

    enum class BlendMode : uint8_t
    {
        Opaque        = 0,
        AlphaBlendF2B = 1
    };

    HD struct Fragment
    {
        float4 color;
        int layer;
        int3 padding; // Padding to make the struct size a multiple of 16 bytes
    };
    static_assert(sizeof(Fragment) % 16 == 0, "Fragment should be aligned to 16 bytes for optimal GPU access.");

    struct PipelineConfig
    {
        int width  = 0;
        int height = 0;

        BlendMode blendMode = BlendMode::Opaque;
        int pointSize       = 1;
        int lineWidth       = 1;

        bool wireframe = false;

        torch::Tensor outputRT;

        PrimitiveType primitiveType = PrimitiveType::Triangles;
    };

    class Rasterizer2D
    {
    public:
        Rasterizer2D();
        ~Rasterizer2D();

        void createPipeline(const PipelineConfig& config, int index);

        void addVertexBuffer(const torch::Tensor& vertexPositions, const torch::Tensor& vertexColors, const torch::Tensor& vertexTexcoords, const torch::Tensor& vertexLayers, int vertexCount, int index);

        void addIndexBuffer(const torch::Tensor& indexBuffer, int indexCount, int index);

        void setClear(const torch::Tensor& clearColor, int clearLayer);

        void bindVertexBufferObject(int index);

        void bindPipeline(int index);

        void drawIndexed(bool clear = true);

        static const int MAX_PIPELINES = 4;
        static const int MAX_VBOS      = 8;

    private:
        void assemblePrimitives();
        void rasterizeTiled();

        PipelineConfig mConfigs[MAX_PIPELINES];

        int mEnabledPipeline = -1;

        struct VertexBufferObject
        {
            torch::Tensor vertexInPositions; // float2
            torch::Tensor vertexInColors;    // float4
            torch::Tensor vertexInTexcoords; // float2
            torch::Tensor vertexInLayers;    // int

            torch::Tensor indexBuffer; // int

            int vertexCount    = 0;
            int indexCount     = 0;
            int primitiveCount = 0;

            torch::Tensor vertexOutPositions; // float2
            torch::Tensor vertexOutColors;    // float4
            torch::Tensor vertexOutTexcoords; // float2
            torch::Tensor vertexOutLayers;    // int

            torch::Tensor primitivePositions; // float2[3]
            torch::Tensor primitiveColors;    // float4[3]
            torch::Tensor primitiveTexcoords; // float2[3]
            torch::Tensor primitiveLayers;    // int[3]
            torch::Tensor primitiveAABBs;     // float4 (min.x, min.y, max.x, max.y)

            torch::Tensor tilesTouched; // int
            torch::Tensor depths;       // int (layer)
        };

        int mEnabledVBO = -1;

        struct RasterizerData
        {
            VertexBufferObject vertexBufferObjects[MAX_VBOS];

            torch::Tensor tileRanges; // int2

            torch::Tensor tileLookupKeys;   // int64_t
            torch::Tensor tileLookupValues; // int
            int64_t tileLookupCapacity = 0;

            // Fragments
            torch::Tensor depthBufferColor; // float4 (RGBA)
            torch::Tensor depthBufferLayer; // int (layer)

            torch::Tensor clearColor = torch::tensor({ 0.0f, 0.0f, 0.0f, 1.0f });
            int clearLayer           = 0;
        } mData;
    };
} // namespace sea_anemones