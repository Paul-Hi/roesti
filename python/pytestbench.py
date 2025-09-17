import time
import sys
import matplotlib
import matplotlib.pyplot as plt
import imageio
import cv2

# sys.path.append('../build/debug/lib')  # adjust as needed
sys.path.append('../build/release/lib')  # adjust as needed


def main():
    import torch
    import pyroesti

    print("pyroesti version:", pyroesti.__version__)

    resolution = (1920, 1080)

    vertex_positions = torch.tensor([
        [-0.6,  0.6],
        [0.0,  0.6],
        [0.0,  0.0],
        [-0.6,  0.0],
        [0.2,  0.4],
        [0.8,  0.4],
        [0.8, -0.2],
        [0.2, -0.2],
    ], dtype=torch.float32).cuda()

    vertex_colors = torch.tensor([
        [1.0, 0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
    ], dtype=torch.float32).cuda()

    vertex_texcoords = torch.zeros((8, 2), dtype=torch.float32).cuda()
    vertex_layers = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32).cuda()
    index_buffer = torch.tensor([
        0, 1, 3, 2,
        2, 4,
        4, 5, 7, 6,
    ], dtype=torch.int32).cuda()
    background_color = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], dtype=torch.float32).cuda()
    output_rt = torch.zeros(
        (resolution[1], resolution[0], 4), dtype=torch.float32, pin_memory=True).cuda()

    config = pyroesti.PipelineConfig(width=resolution[0],
                                     height=resolution[1],
                                     blend_mode=pyroesti.BlendMode.ALPHA_BLEND_F2B,
                                     point_size=1,
                                     line_width=1,
                                     wireframe=False,
                                     output_rt=output_rt,
                                     primitive_type=pyroesti.PrimitiveType.TRIANGLE_STRIP)

    rasterizer = pyroesti.Rasterizer2D()
    rasterizer.create_pipeline(config, 0)
    rasterizer.add_vertex_buffer(vertex_positions, vertex_colors,
                                 vertex_texcoords, vertex_layers, vertex_positions.shape[0], 0)
    rasterizer.add_index_buffer(index_buffer, index_buffer.shape[0], 0)
    rasterizer.set_clear(background_color, 0)

    cv2.namedWindow("Pyroesti Testbench", cv2.WINDOW_NORMAL)

    start_time = time.time()
    while True:
        dt = time.time() - start_time
        start_time = time.time()
        rasterizer.bind_pipeline(0)
        rasterizer.bind_vertex_buffer_object(0)
        rasterizer.draw_indexed()

        img_np = output_rt.to('cpu').numpy()

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA)

        cv2.imshow("Pyroesti Testbench", img_bgr)

        # update vertices so that quads rotate
        delta_angle = torch.tensor(dt, dtype=torch.float32).cuda()
        cos_a = torch.cos(delta_angle)
        sin_a = torch.sin(delta_angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=torch.float32).cuda()

        quad_centers = [
            vertex_positions[0:4].mean(dim=0, keepdim=True),
            vertex_positions[4:8].mean(dim=0, keepdim=True)
        ]

        vertex_positions[0:4] = torch.matmul(
            vertex_positions[0:4] - quad_centers[0], rotation_matrix.T
        ) + quad_centers[0]
        vertex_positions[4:8] = torch.matmul(
            vertex_positions[4:8] - quad_centers[1], rotation_matrix.T
        ) + quad_centers[1]

        key = cv2.waitKey(1)
        if key == ord('q') or cv2.getWindowProperty("Pyroesti Testbench", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
