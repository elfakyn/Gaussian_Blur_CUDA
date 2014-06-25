#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>

using std::cout;
using std::fstream;

int dummy; // makes it work

#define BLOCK_SIZE_MULTIPLIER 4 // TODO: make ths dynamic as to not exceed 1024 blocks
#define BLOCK_SIZE_LINEAR 2000

#define CLAMP(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define DIVCEIL(x, y) (((x) + (y) - 1) / (y)) // division that rounds up

__global__ void separateChannels(uchar4 *inputRGBA,
                                 int *_x_dim,
                                 int *_y_dim,
                                 unsigned char *r,
                                 unsigned char *g,
                                 unsigned char *b)
{
    int x_dim = *_x_dim;
    int y_dim = *_y_dim;

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= x_dim * y_dim) {
        return;  // value out of bounds, don't do anything
    }
    uchar4 tmp = inputRGBA[i];
    r[i] = (unsigned char)tmp.x;
    g[i] = (unsigned char)tmp.y;
    b[i] = (unsigned char)tmp.z;
}

__global__ void recombineChannels(unsigned char *r,
                                  unsigned char *g,
                                  unsigned char *b,
                                  int *_x_dim,
                                  int *_y_dim,
                                  uchar4 *outputRGBA)
{
    int x_dim = *_x_dim;
    int y_dim = *_y_dim;

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= x_dim * y_dim) {
        return;
    }
    uchar4 tmp;
    tmp.x = r[i];
    tmp.y = g[i];
    tmp.z = b[i];

    tmp.w = 255;  // no transparency
    outputRGBA[i] = tmp;
}

__global__ void gaussianBlur(unsigned char *in,
                             float *filter,
                             int *_filter_dim,
                             int *_x_dim,
                             int *_y_dim,
                             unsigned char *out)
{
    int filter_dim = *_filter_dim;
    int x_dim = *_x_dim;
    int y_dim = *_y_dim;

    int x_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int y_pos = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_pos >= x_dim || y_pos >= y_dim) {
        return;  // out of bounds
    }

    float val = 0.0f;

    int x_i_pos, y_i_pos, offset = filter_dim / 2;
    for (int x_i = 0; x_i < filter_dim; x_i++) {
        for (int y_i = 0; y_i < filter_dim; y_i++) {
            // multiply each value in the adjacent pixels clamped to the edges
            // of the image by the corresponding filter value and add it to the
            // total value that will be set in the blurred image
            x_i_pos = x_i + x_pos - offset;
            y_i_pos = y_i + y_pos - offset;
            val += float(in[CLAMP(x_i_pos, 0, x_dim - 1) +
                            CLAMP(y_i_pos, 0, y_dim - 1) * x_dim]) *
                filter[x_i + y_i * filter_dim];
        }
    }
    out[x_pos + y_pos * x_dim] = (unsigned char)val;
}

int main()
{
    // initialize, allocate and read relevant host image variables
    int h_x_dim, h_y_dim;
    uchar4 *h_img;

    fstream h_img_stream_in;
    h_img_stream_in.open("test.ppm", fstream::in);
    h_img_stream_in.ignore(2, EOF); // ignores the P3 at the beginning of the file
    h_img_stream_in >> h_x_dim >> h_y_dim;
    h_img_stream_in >> dummy;
    
    int h_xy_dim = h_x_dim * h_y_dim;
    h_img = (uchar4 *) malloc(h_xy_dim * sizeof(uchar4));
    
    int x, y, z;
    for (int i = 0; i < h_xy_dim; i++) {
        h_img_stream_in >> x >> y >> z;
        h_img[i].x = (unsigned char)x;
        h_img[i].y = (unsigned char)y;
        h_img[i].z = (unsigned char)z;
        h_img[i].w = 255;
    }

    h_img_stream_in.close();

    // initialize, allocate and read relevant host filter variables
    int h_filter_dim;
    float *h_filter;
    fstream h_filter_stream_in;
    h_filter_stream_in.open("filter", fstream::in);
    h_filter_stream_in >> h_filter_dim;

    int h_filter_dim2 = h_filter_dim * h_filter_dim;
    h_filter = (float *) malloc(h_filter_dim2 * sizeof(float));

    for (int i = 0; i < h_filter_dim2; i++) {
        h_filter_stream_in >> h_filter[i];
    }

    h_filter_stream_in.close();

    // initialie and allocate relevant device variables
    uchar4 *d_img_in, *d_img_out;
    unsigned char *d_img_r_in, *d_img_g_in, *d_img_b_in;
    unsigned char *d_img_r_out, *d_img_g_out, *d_img_b_out;
    int *d_x_dim, *d_y_dim;

    float *d_filter;
    int *d_filter_dim;

    cudaMalloc((void **) &d_img_in, h_xy_dim * sizeof(uchar4));
    cudaMalloc((void **) &d_img_out, h_xy_dim * sizeof(uchar4));
    cudaMalloc((void **) &d_img_r_in, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_img_g_in, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_img_b_in, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_img_r_out, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_img_g_out, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_img_b_out, h_xy_dim * sizeof(unsigned char));
    cudaMalloc((void **) &d_x_dim, sizeof(int));
    cudaMalloc((void **) &d_y_dim, sizeof(int));
    cudaMalloc((void **) &d_filter, h_filter_dim2 * sizeof(float));
    cudaMalloc((void **) &d_filter_dim, sizeof(int));

    cudaMemcpy(d_img_in, h_img, h_xy_dim * sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_dim, &h_x_dim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_dim, &h_y_dim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, h_filter_dim2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter_dim, &h_filter_dim, sizeof(int), cudaMemcpyHostToDevice);

    // determine appropriate block dimensions and numbers
    dim3 block_dim = dim3(h_filter_dim * BLOCK_SIZE_MULTIPLIER,
                          h_filter_dim * BLOCK_SIZE_MULTIPLIER, 1);
    dim3 block_number = dim3(DIVCEIL(h_x_dim, block_dim.x),
                             DIVCEIL(h_y_dim, block_dim.y), 1);

    cout<<"starting operations\n";
    // perform operations on GPU

    separateChannels<<<BLOCK_SIZE_LINEAR, DIVCEIL(h_xy_dim, BLOCK_SIZE_LINEAR)>>>
        (d_img_in, d_x_dim, d_y_dim, d_img_r_in, d_img_g_in, d_img_b_in);
    cout<<"channels separated\n";
    
    gaussianBlur<<<block_dim, block_number>>>
        (d_img_r_in, d_filter, d_filter_dim, d_x_dim, d_y_dim, d_img_r_out);
    cout<<"red blurred\n";
    gaussianBlur<<<block_dim, block_number>>>
        (d_img_g_in, d_filter, d_filter_dim, d_x_dim, d_y_dim, d_img_g_out);
    cout<<"green blurred\n";
    gaussianBlur<<<block_dim, block_number>>>
        (d_img_b_in, d_filter, d_filter_dim, d_x_dim, d_y_dim, d_img_b_out);
    cout<<"blue blurred\n";
    
    recombineChannels<<<BLOCK_SIZE_LINEAR, DIVCEIL(h_xy_dim, BLOCK_SIZE_LINEAR)>>>
        (d_img_r_out, d_img_g_out, d_img_b_out, d_x_dim, d_y_dim, d_img_out);
    cout<<"channels recombined\n";
    cout<<"operations done.\n";

    // copy data back from GPU and print it to file
    cudaMemcpy(h_img, d_img_out, h_xy_dim * sizeof(uchar4), cudaMemcpyDeviceToHost);

    fstream h_img_stream_out;
    h_img_stream_out.open("blurred.ppm", fstream::out);
    h_img_stream_out << "P3 " << h_x_dim << " " << h_y_dim << "\n255\n";
    for (int i = 0; i < h_xy_dim; i++) {
        h_img_stream_out << (int)h_img[i].x << " " << (int)h_img[i].y << " " << (int)h_img[i].z << "\n";
    }
    h_img_stream_out.close();

    // free memory and exit
    free(h_img);
    free(h_filter);

    cudaFree(d_img_in);
    cudaFree(d_img_out);
    cudaFree(d_img_r_in);
    cudaFree(d_img_g_in);
    cudaFree(d_img_b_in);
    cudaFree(d_img_r_out);
    cudaFree(d_img_g_out);
    cudaFree(d_img_b_out);
    cudaFree(d_x_dim);
    cudaFree(d_y_dim);
    cudaFree(d_filter);
    cudaFree(d_filter_dim);

    return 0;
}