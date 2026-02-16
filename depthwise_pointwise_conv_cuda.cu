
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void depthwise_pointwise_conv(
    const float* input, 
    const float* depthwise_weights,
    const float* pointwise_weights,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int height, int width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int output_height,
    int output_width)
{
    int batch_idx = blockIdx.x / ((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE * (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int block_y = (blockIdx.x % (((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE))) / ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int block_x = (blockIdx.x % (((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE))) % ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int y_start = block_y * BLOCK_SIZE;
    int x_start = block_x * BLOCK_SIZE;

    int out_channel = threadIdx.x;

    __shared__ float depthwise_results[BLOCK_SIZE * BLOCK_SIZE][in_channels];

    for (int out_y = y_start; out_y < y_start + BLOCK_SIZE && out_y < output_height; out_y++) {
        for (int out_x = x_start; out_x < x_start + BLOCK_SIZE && out_x < output_width; out_x++) {
            int in_y_base = out_y * stride - padding;
            int in_x_base = out_x * stride - padding;

            if (threadIdx.x < in_channels) {
                float sum = 0;
                for (int ki = 0; ki < kernel_size; ki++) {
                    for (int kj = 0; kj < kernel_size; kj++) {
                        int y = in_y_base + dilation * ki;
                        int x = in_x_base + dilation * kj;
                        if (y >= 0 && y < height && x >= 0 && x < width) {
                            int input_idx = batch_idx * in_channels * height * width + threadIdx.x * height * width + y * width + x;
                            int weight_idx = threadIdx.x * kernel_size * kernel_size + ki * kernel_size + kj;
                            sum += input[input_idx] * depthwise_weights[weight_idx];
                        }
                    }
                }
                int pos_y = out_y - y_start;
                int pos_x = out_x - x_start;
                depthwise_results[pos_y * BLOCK_SIZE + pos_x][threadIdx.x] = sum;
            }
            __syncthreads();

            if (out_channel < out_channels) {
                float psum = 0;
                for (int c = 0; c < in_channels; c++) {
                    int pos_y = out_y - y_start;
                    int pos_x = out_x - x_start;
                    psum += depthwise_results[pos_y * BLOCK_SIZE + pos_x][c] * pointwise_weights[out_channel * in_channels + c];
                }
                if (bias) {
                    psum += bias[out_channel];
                }
                int output_idx = batch_idx * out_channels * output_height * output_width +
                                 out_channel * output_height * output_width +
                                 out_y * output_width + out_x;
                output[output_idx] = psum;
            }
            __syncthreads();
        }
    }
}

torch::Tensor depthwise_pointwise_conv_cuda(torch::Tensor input, 
                                            torch::Tensor depthwise_weights,
                                            torch::Tensor pointwise_weights,
                                            torch::Tensor bias_tensor,
                                            int stride, int padding, int dilation) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int out_channels = pointwise_weights.size(0);
    int kernel_size = depthwise_weights.size(2);

    int output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width}, input.options());

    int threads = out_channels;
    int num_blocks_per_sample = ((output_height + BLOCK_SIZE - 1) / BLOCK_SIZE) * ((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int num_blocks = batch_size * num_blocks_per_sample;

    const float* bias_ptr = bias_tensor.data_ptr<float>();
    if (bias_tensor.numel() == 0) {
        bias_ptr = nullptr;
    }

    depthwise_pointwise_conv<<<num_blocks, threads>>>(
        input.data_ptr<float>(),
        depthwise_weights.data_ptr<float>(),
        pointwise_weights.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels,
        height, width,
        kernel_size, stride, padding, dilation,
        output_height, output_width
    );

    return output;
}
