#ifdef _WIN32
  #include <GL/gl3w.h>
#else
  #include <GL/glew.h>
#endif
#include <GLFW/glfw3.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/common.h>
#include <Eigen/Dense>

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/dlss.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/npy.hpp>

void simple_glfw_error_callback(int error, const char* description) 
{
    std::cout << "GLFW error #" << error << ": " << description << std::endl;
}

__global__ void dlss_prep_kernel(
	Eigen::Vector2i resolution,
	float* depth_buffer,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x() || y >= resolution.y()) {
		return;
	}

	uint32_t idx = x + resolution.x() * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	Eigen::Vector2f mvec = {0., 0.}; // motion vector

	surf2Dwrite(make_float2(mvec.x(), mvec.y()), mvec_surface, x_orig * sizeof(float2), y_orig);

	// Scale depth buffer to be guaranteed in [0,1].
	surf2Dwrite(std::min(std::max(depth / 128.0f, 0.0f), 1.0f), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

void render_frame(ngp::CudaRenderBuffer& render_buffer) 
{
    std::cout << "render frame begin" << std::endl;
    
    // CUDA stuff
	tcnn::StreamAndEvent m_stream;
    render_buffer.clear_frame(m_stream.get());
    render_buffer.set_color_space(ngp::EColorSpace::Linear);
	render_buffer.set_tonemap_curve(ngp::ETonemapCurve::Identity);

    const std::string path{"depth.npy"};
    std::cout << "load depth buffer..." << std::endl;
	std::vector<float> data;
	std::vector<unsigned long> shape;
	bool is_fortran;
	npy::LoadArrayFromNumpy(path, shape, is_fortran, data);
    std::cout << "buffer size: " << data.size() << std::endl;
	render_buffer.host_to_depth_buffer(data);

    // Prepare DLSS data: motion vectors, scaled depth, exposure
    std::cout << "prepare the dlss data..." << std::endl;
    auto res = render_buffer.in_resolution();
    //bool distortion = false;
    const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { tcnn::div_round_up((uint32_t)res.x(), threads.x), tcnn::div_round_up((uint32_t)res.y(), threads.y), 1 };
    float m_dlss_sharpening = 0.0;
    dlss_prep_kernel<<<blocks, threads, 0, m_stream.get()>>>(
			res,
			render_buffer.depth_buffer(),
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure()
	);
    render_buffer.set_dlss_sharpening(m_dlss_sharpening);

    std::cout << "run dlss..." << std::endl;
    float m_exposure = 0.0;
    Eigen::Array4f m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};
    render_buffer.accumulate(m_exposure, m_stream.get());
    render_buffer.tonemap(m_exposure, m_background_color, ngp::EColorSpace::Linear, m_stream.get());
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}

int main() 
{
    std::cout << "custom glfw init" << std::endl;
    glfwSetErrorCallback(simple_glfw_error_callback);
    if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}
    std::cout << "custom enable dlss" << std::endl;
    try {
		ngp::vulkan_and_ngx_init();
	} catch (const std::runtime_error& e) {
		tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
	}

    int in_height = 640;
	int in_width = 380;
    ngp::CudaRenderBuffer m_windowless_render_surface{std::make_shared<ngp::CudaSurface2D>()};
    m_windowless_render_surface.resize({in_width, in_height});
	m_windowless_render_surface.reset_accumulation();

    unsigned long out_height = 1080;
	unsigned long out_width = 1920;
    // enable dlss
	tlog::info() << "custom enable dlss for render buffer";
	m_windowless_render_surface.enable_dlss({out_width, out_height});
	auto render_res = m_windowless_render_surface.in_resolution();
	if (m_windowless_render_surface.dlss()) {
		render_res = m_windowless_render_surface.dlss()->clamp_resolution(render_res);
	}
	m_windowless_render_surface.resize(render_res);

    render_frame(m_windowless_render_surface);

    std::cout << "begin to transfer data..." << std::endl;

    //float *result = (float*)malloc(sizeof(float)*out_height*out_width*4);
    std::vector<float> result(out_height*out_width*4, 0.0);

    cudaError_t x = cudaMemcpy2DFromArray(&result[0], out_width * sizeof(float) * 4, m_windowless_render_surface.surface_provider().array(), 0, 0, out_width * sizeof(float) * 4, out_height, cudaMemcpyDeviceToHost);
    CUDA_CHECK_THROW(x);

    const std::vector<long unsigned> shape{out_height, out_width, 4};
	const bool fortran_order{false};
    const std::string path{"out.npy"};
	
	// try to save frame_buffer here?
	std::cout << "save frame buffer..." << std::endl;
	npy::SaveArrayAsNumpy(path, fortran_order, shape.size(), shape.data(), result);

    //CUDA_CHECK_THROW(cudaMemcpy2DToArray(m_windowless_render_surface.surface_provider().array(), 0, 0, result, out_width * sizeof(float) * 4, out_width * sizeof(float) * 4, out_height, cudaMemcpyDeviceToHost));
    //free(result);

    return 0;
}