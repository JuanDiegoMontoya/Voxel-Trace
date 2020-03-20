// taken from
// https://stackoverflow.com/questions/20762828/crash-with-cuda-ogl-interop/20765755#20765755
#include "stdafx.h"
#include "testCU.h"

#include "CommonDevice.cuh"
#include "cuda_gl_interop.h"
#include <vector_types.h>
#include <vao.h>
#include <vbo.h>
#include <shader.h>

const unsigned int window_width = 512;
const unsigned int window_height = 512;

GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;

static VAO* vao;
static VBO* vbo;

void initGLandCUDA() {
  //int argc = 0;
  //char** argv = NULL;
  //glutInit(&argc, argv);
  //glutInitDisplayMode(GLUT_RGBA);
  //glutInitWindowSize(window_width, window_height);
  //glutCreateWindow("CUDA GL Interop");

  //glewInit();

  //glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &viewGLTexture);
  glBindTexture(GL_TEXTURE_2D, viewGLTexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, window_width, window_height, 0, GL_RGBA, GL_FLOAT, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  //cudaCheck(cudaGLSetGLDevice(0));
  cudaCheck(cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));


  float quadVertices[] =
  {
    // positions        // texture Coords
    -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
     1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
     1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
  };

  // setup screen texture pointers
  vbo = new VBO(&quadVertices[0],
    sizeof(quadVertices), GL_STATIC_DRAW);
  VBOlayout layout;
  layout.Push<float>(3); // pos
  layout.Push<float>(2); // texcoord
  vao = new VAO();
  vao->AddBuffer(*vbo, layout);
}


__global__ void renderingKernel(cudaSurfaceObject_t image, float time) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  //uchar4 color = make_uchar4(x / 2, y / 2, 0, 127);
  float4 color = make_float4(x / 512.f, y / 512.f, 0, .5f);
  color.z = cos(time);
  //float3 color = make_float3(x / 512.f, y / 512.f, 0);
  surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}


void callCUDAKernel(cudaSurfaceObject_t image) {
  dim3 block(256, 1, 1);
  dim3 grid(2, 512, 1);
  renderingKernel<<<grid, block>>>(image, glfwGetTime());
  cudaCheck(cudaPeekAtLastError());
  cudaCheck(cudaDeviceSynchronize());
}

void RenderTestCUDA() {
  cudaCheck(cudaGraphicsMapResources(1, &viewCudaResource));

  cudaArray_t viewCudaArray;
  cudaCheck(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));

  cudaResourceDesc viewCudaArrayResourceDesc;
  memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
  viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
  viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

  cudaSurfaceObject_t viewCudaSurfaceObject;
  cudaCheck(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

  callCUDAKernel(viewCudaSurfaceObject);

  cudaCheck(cudaDestroySurfaceObject(viewCudaSurfaceObject));

  cudaCheck(cudaGraphicsUnmapResources(1, &viewCudaResource));

  cudaCheck(cudaStreamSynchronize(0));

  //glBindTexture(GL_TEXTURE_2D, viewGLTexture);
  //{
  //  glBegin(GL_QUADS);
  //  {
  //    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
  //    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
  //    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
  //    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
  //  }
  //  glEnd();
  //}
  //glBindTexture(GL_TEXTURE_2D, 0);
  //glFinish();
  // draw fullscreen quad
  ShaderPtr s = Shader::shaders["fullscreen"];
  s->Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, viewGLTexture);
  s->setInt("tex", 0);
  vao->Bind();
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  vao->Unbind();
  s->Unuse();
}

void InitTestCUDA()
{
  initGLandCUDA();

  //glutDisplayFunc(renderFrame);
  //glutKeyboardFunc(keyboard);
  //glutMouseFunc(mouse);
  //glutMainLoop();
}