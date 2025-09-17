#include "roesti.hpp"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace py = pybind11;
PYBIND11_MODULE(pyroesti, m)
{
    m.doc()               = "Python bindings for the roesti rasterizer library";
    m.attr("__version__") = "0.0.1";
    m.attr("__author__")  = "Paul Himmler";
    m.attr("__license__") = "Apache-2.0";
    py::enum_<roesti::PrimitiveType>(m, "PrimitiveType", "Type of geometric primitive to rasterize")
        .value("POINTS", roesti::PrimitiveType::Points, "Render points")
        .value("LINES", roesti::PrimitiveType::Lines, "Render lines")
        .value("LINE_STRIP", roesti::PrimitiveType::LineStrip, "Render connected lines")
        .value("TRIANGLES", roesti::PrimitiveType::Triangles, "Render triangles")
        .value("TRIANGLE_STRIP", roesti::PrimitiveType::TriangleStrip, "Render connected triangles")
        .export_values();

    py::enum_<roesti::BlendMode>(m, "BlendMode", "Blending mode for rasterization")
        .value("OPAQUE", roesti::BlendMode::Opaque, "No blending, opaque rendering")
        .value("ALPHA_BLEND_F2B", roesti::BlendMode::AlphaBlendF2B, "Alpha blending front-to-back")
        .export_values();

    py::class_<roesti::Fragment>(m, "Fragment", "Fragment data for rasterization")
        .def(py::init<>(), "Create a default Fragment")
        .def_property("color", [](const roesti::Fragment& self)
                      { return self.color; }, [](roesti::Fragment& self, const decltype(roesti::Fragment::color)& v)
                      { self.color = v; }, "Fragment color (float4)")
        .def_property("layer", [](const roesti::Fragment& self)
                      { return self.layer; }, [](roesti::Fragment& self, int v)
                      { self.layer = v; }, "Fragment layer");

    py::class_<roesti::PipelineConfig>(m, "PipelineConfig", "Configuration for a rasterizer pipeline")
        .def(py::init<>(), "Create a default pipeline configuration")
        .def(py::init<
                 int, int, roesti::BlendMode, int, int, bool, torch::Tensor, roesti::PrimitiveType>(),
             py::arg("width")          = 0,
             py::arg("height")         = 0,
             py::arg("blend_mode")     = roesti::BlendMode::Opaque,
             py::arg("point_size")     = 1,
             py::arg("line_width")     = 1,
             py::arg("wireframe")      = false,
             py::arg("output_rt")      = torch::Tensor(),
             py::arg("primitive_type") = roesti::PrimitiveType::Triangles,
             "Create a pipeline configuration with all properties")
        .def_property("width", [](const roesti::PipelineConfig& self)
                      { return self.width; }, [](roesti::PipelineConfig& self, int v)
                      { self.width = v; }, "Width of the output render target")
        .def_property("height", [](const roesti::PipelineConfig& self)
                      { return self.height; }, [](roesti::PipelineConfig& self, int v)
                      { self.height = v; }, "Height of the output render target")
        .def_property("blend_mode", [](const roesti::PipelineConfig& self)
                      { return self.blendMode; }, [](roesti::PipelineConfig& self, roesti::BlendMode v)
                      { self.blendMode = v; }, "Blending mode")
        .def_property("point_size", [](const roesti::PipelineConfig& self)
                      { return self.pointSize; }, [](roesti::PipelineConfig& self, int v)
                      { self.pointSize = v; }, "Size of points")
        .def_property("line_width", [](const roesti::PipelineConfig& self)
                      { return self.lineWidth; }, [](roesti::PipelineConfig& self, int v)
                      { self.lineWidth = v; }, "Width of lines")
        .def_property("wireframe", [](const roesti::PipelineConfig& self)
                      { return self.wireframe; }, [](roesti::PipelineConfig& self, bool v)
                      { self.wireframe = v; }, "Render in wireframe mode")
        .def_property("output_rt", [](const roesti::PipelineConfig& self)
                      { return self.outputRT; }, [](roesti::PipelineConfig& self, const torch::Tensor& v)
                      { self.outputRT = v; }, "Output render target tensor (should be float32 tensor of shape (H, W, 4))")
        .def_property("primitive_type", [](const roesti::PipelineConfig& self)
                      { return self.primitiveType; }, [](roesti::PipelineConfig& self, roesti::PrimitiveType v)
                      { self.primitiveType = v; }, "Type of primitive to rasterize");

    py::class_<roesti::Rasterizer2D>(m, "Rasterizer2D", "2D rasterizer for rendering primitives")
        .def(py::init<>(), "Create a new Rasterizer2D instance")
        .def("create_pipeline",
             &roesti::Rasterizer2D::createPipeline,
             py::arg("config"),
             py::arg("index"),
             "Create a rendering pipeline with the given configuration and index")
        .def("add_vertex_buffer", [](roesti::Rasterizer2D& self, torch::Tensor vertex_positions, torch::Tensor vertex_colors, torch::Tensor vertex_texcoords, torch::Tensor vertex_layers, int vertex_count, int index)
             { self.addVertexBuffer(vertex_positions, vertex_colors, vertex_texcoords, vertex_layers, vertex_count, index); }, py::arg("vertex_positions"), py::arg("vertex_colors"), py::arg("vertex_texcoords"), py::arg("vertex_layers"), py::arg("vertex_count"), py::arg("index"), "Add a vertex buffer to the rasterizer")
        .def("add_index_buffer", [](roesti::Rasterizer2D& self, torch::Tensor index_buffer, int index_count, int index)
             { self.addIndexBuffer(index_buffer, index_count, index); }, py::arg("index_buffer"), py::arg("index_count"), py::arg("index"), "Add an index buffer to the rasterizer")
        .def("set_clear", [](roesti::Rasterizer2D& self, torch::Tensor clear_color, int clear_layer)
             { self.setClear(clear_color, clear_layer); }, py::arg("clear_color"), py::arg("clear_layer"), "Set the clear color and layer for the render target")
        .def("bind_vertex_buffer_object", &roesti::Rasterizer2D::bindVertexBufferObject, py::arg("index"), "Bind a vertex buffer object by its index")
        .def("bind_pipeline", &roesti::Rasterizer2D::bindPipeline, py::arg("index"), "Bind a pipeline by its index")
        .def("draw_indexed", &roesti::Rasterizer2D::drawIndexed, py::arg("clear") = true, "Draw indexed primitives, optionally clearing before draw")
        .def_property_readonly_static("MAX_PIPELINES", [](py::object)
                                      { return roesti::Rasterizer2D::MAX_PIPELINES; }, "Maximum number of pipelines supported")
        .def_property_readonly_static("MAX_VBOS", [](py::object)
                                      { return roesti::Rasterizer2D::MAX_VBOS; }, "Maximum number of vertex buffer objects supported");

    // Note: VertexBufferObject and RasterizerData are private, so not exposed.
}
