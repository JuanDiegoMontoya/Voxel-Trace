#include "stdafx.h"
#include "Voxtrace.h"
#include "RayCamera.h"

#include "Renderer.h"
#include <Engine.h>
#include <Pipeline.h>
#include <camera.h>
#include <Line.h>
#include <shader.h>

#include <vbo.h>
#include <vao.h>

#include "CommonDevice.cuh"
#include "cuda_gl_interop.h"
#include "pick.h"

surface<void, 2> screenSurface;

__device__ static bool swagCallback(glm::vec3 p, Voxels::Block* block, glm::vec3 norm)
{
	if (block)
	{
		//printf("hit pos: %.0f, %.0f, %.0f\n", p.x, p.y, p.z);
		return true;
	}
	return false;
}

__global__ static void epicRayTracer(Voxels::Block* pWorld, glm::ivec3 worldDim,
	PerspectiveRayCamera cam, 
	glm::vec3 chunkDim, glm::vec2 imgSize, float time)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = imgSize.x * imgSize.y;

	//printf("index = %d, stride = %d, n = %d\n", index, stride, n);
	for (int i = index; i < n; i += stride)
	{
		glm::vec2 imgPos = expand(i, imgSize.y);
		//glm::vec2 imgPos(x, y);
		glm::vec2 screenCoord(
			(2.0f * imgPos.x) / imgSize.x - 1.0f,
			(-2.0f * imgPos.y) / imgSize.y + 1.0f);
		Ray ray = cam.makeRay(screenCoord);
		glm::vec4 val = glm::vec4(ray.direction * .5f + glm::vec3(1), 1);

		val = { .53f, .81f, .92f, 1 };
		auto cb = [&val](glm::vec3 p, Voxels::Block* block, glm::vec3 norm)->bool
		{
			if (block)
			{
				//if (block->alpha == 0)
				//	return false;
				//printf("hit pos: %.0f, %.0f, %.0f\n", p.x, p.y, p.z);
				//val = glm::vec4(block->diffuse, 1.f);
				val = glm::vec4((norm + glm::vec3(1)) * .5f, 1.f);
				//val = glm::vec4(0, 0, 0, 1);
				return true;
			}
			return false;
		};

		raycast(pWorld, worldDim, ray.origin, ray.direction, 150.f, cb);

		// write final pixel value
		surf2Dwrite(val, screenSurface, imgPos.x * sizeof(val), imgSize.y - imgPos.y - 1);
	}
}

namespace Voxels
{
	namespace
	{
		PerspectiveRayCamera cam;
		LinePool* lines = nullptr;

		// world description
		Block* blocks = nullptr;
		glm::ivec3 chunkDim = { 10, 10, 10 };
		int numBlocks = chunkDim.x * chunkDim.y * chunkDim.z;

		// screen info
		glm::vec2 screenDim = { 500, 265 };
		//glm::vec2 screenDim = { 1920, 1080 };
		
		// rendering shiz
		VBO* vbo = nullptr;
		VAO* vao = nullptr;
		GLuint screenTexture = -1;

		// cuda GL stuff
		cudaGraphicsResource* imageResource = nullptr;
		cudaArray* arr = nullptr;

		const int KernelBlockSize = 256;
		const int KernelNumBlocks = (screenDim.x * screenDim.y + KernelBlockSize - 1) / KernelBlockSize;
	}

	void Init()
	{
		Engine::PushRenderCallback(Render, 4);
		InitGLStuff();
		InitBlocks();
	}

	void InitGLStuff()
	{
		// TODO: move this to Vertices.h or something
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

		// generate screen texture memory
		glGenTextures(1, &screenTexture);
		glBindTexture(GL_TEXTURE_2D, screenTexture);
		// cuda behavior becomes extremely weird when using RGB textures with it
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
			screenDim.x, screenDim.y, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		glBindTexture(GL_TEXTURE_2D, 0);

		// init texture color
		glm::vec4 defColor(1, .1, .1, 1.);
		glClearTexImage(screenTexture, 0, GL_RGBA, GL_FLOAT, &defColor[0]);

		// register the texture as a cuda resource
		cudaCheck(
			cudaGraphicsGLRegisterImage(&imageResource, screenTexture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void InitBlocks()
	{
		// shared memory so that the GPU and CPU can both read and write blocks
		cudaCheck(cudaMallocManaged(&blocks, numBlocks * sizeof(Block)));

		for (int i = 0; i < numBlocks; i++)
		{
			auto pos = expand(i, chunkDim.y);
			if (pos.x > 3 && pos.x < 7 &&
				pos.y > 3 && pos.y < 7)
			{
				//blocks[i].diffuse = { 1, 0, 0 };
			}
			blocks[i].diffuse = Utils::get_random_vec3_r(0, 1);
			blocks[i].alpha = 1;
		}
	}

	void Render()
	{
		auto c = Renderer::GetPipeline()->GetCamera(0);
		cam = PerspectiveRayCamera(c->GetPos(), c->GetPos() + c->GetDir(), 
			glm::vec3(0, 1, 0), glm::radians(30.f), screenDim.x / screenDim.y);

		// ray trace her
		{
			cudaCheck(cudaGraphicsMapResources(1, &imageResource, 0));
			cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0));
			cudaCheck(cudaBindSurfaceToArray(screenSurface, arr));

			//printf("screenDim = %f, %f\n", screenDim.x, screenDim.y);
			epicRayTracer<<<KernelNumBlocks, KernelBlockSize>>>(
				blocks, chunkDim, cam, chunkDim, screenDim, glfwGetTime());
			cudaDeviceSynchronize();

			cudaCheck(cudaGraphicsUnmapResources(1, &imageResource, 0));
		}

		// draw fullscreen quad
		ShaderPtr s = Shader::shaders["fullscreen"];
		s->Use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, screenTexture);
		s->setInt("tex", 0);
		vao->Bind();
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		vao->Unbind();
		s->Unuse();

		glClear(GL_DEPTH_BUFFER_BIT);
		if (lines)
		{
			ShaderPtr s = Shader::shaders["line"];
			s->Use();
			glm::mat4 model(1);
			glm::mat4 view = c->GetView();
			glm::mat4 proj = c->GetProj();
			s->setMat4("u_model", model);
			s->setMat4("u_view", view);
			s->setMat4("u_proj", proj);
			lines->Draw();
		}
	}

	void CameraRaySnapshot()
	{
		delete lines; // ok if null (e.g. first instance)

		std::vector<glm::vec3> poss, dirs, tClrs, bClrs;

		glm::vec2 imgSize(screenDim);
		for (int x = 0; x < imgSize.x; x++)
		{
			for (int y = 0; y < imgSize.y; y++)
			{
				glm::vec2 screenCoord(
					(2.0f * x) / imgSize.x - 1.0f,
					(-2.0f * y) / imgSize.y + 1.0f);
				Ray ray = cam.makeRay(screenCoord);
				poss.push_back(ray.origin);
				dirs.push_back(ray.direction);
			}
		}

		for (int i = 0; i < poss.size(); i++)
		{
			tClrs.push_back(glm::vec3(1));
			bClrs.push_back(glm::vec3(0));
		}

		lines = new LinePool(poss, dirs, tClrs, bClrs);
	}
}
