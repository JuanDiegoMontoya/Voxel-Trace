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
//#include "pick.h"
#include "Randy.h"
#include "TraceInfo.h"
#include "RTKernel.cuh"


namespace Voxels
{
	namespace
	{
		LinePool* lines = nullptr;

		// world description
		Block* blocks = nullptr;
		glm::ivec3 chunkDim = { 10, 10, 10 };
		int numBlocks = chunkDim.x * chunkDim.y * chunkDim.z;

		// screen info
		//glm::vec2 screenDim = { 500, 265 };
		//glm::vec2 screenDim = { 1920, 1080 }; // 1080p
		//glm::vec2 screenDim = { 1280, 720 };  // 720p
		glm::vec2 screenDim = { 853, 480 };   // 480p
		//glm::vec2 screenDim = { 125, 65 };
		TraceInfo info;

		// rendering shiz
		VBO* vbo = nullptr;
		VAO* vao = nullptr;
		GLuint screenTexture = -1;

		// cuda GL stuff
		cudaGraphicsResource* imageResource = nullptr;
		cudaArray* arr = nullptr;

		const int KernelBlockSize = 256;
		const int KernelNumBlocks = (screenDim.x * screenDim.y + KernelBlockSize - 1) / KernelBlockSize;
		curandState_t* states;
	}

	void Init()
	{
		Engine::PushRenderCallback(Render, 4);
		InitGLStuff();
		InitBlocks();
		//InitCUDARand(screenDim.x * screenDim.y);
		InitCUDARand(states, KernelBlockSize * KernelNumBlocks);

		info.imgSize = screenDim;
		info.numShadowRays = 10;
	}

	void Shutdown()
	{
		ShutdownCUDARands(states);
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
			blocks[i].alpha = 0;
			auto pos = expand(i, chunkDim.x, chunkDim.y);
			//if (glm::all(glm::lessThan(pos, { 5, 5, 5 })))
			{
				blocks[i].alpha = rand() % 100 > 80 ? 1 : 0;
				if (rand() % 100 > 95)
					blocks[i].alpha = .5f;
				//blocks[i].diffuse = { 1, 0, 0 };
			}
			blocks[i].diffuse = Utils::get_random_vec3_r(0, 1);
		}
	}

	void Render()
	{
		auto c = Renderer::GetPipeline()->GetCamera(0);
		info.camera = PerspectiveRayCamera(c->GetPos(), c->GetPos() + c->GetDir(), 
			glm::vec3(0, 1, 0), glm::radians(30.f), screenDim.x / screenDim.y);


		// ray trace her
		{
			cudaCheck(cudaGraphicsMapResources(1, &imageResource, 0));
			cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0));
			cudaCheck(cudaBindSurfaceToArray(GetScreenSurface(), arr));

			// passing the info struct in creates crashes when calling info.camera.makeRay
			epicRayTracer<<<KernelNumBlocks, KernelBlockSize>>>(
				blocks, chunkDim, info.camera, info.numShadowRays, info.imgSize,
				chunkDim, *Renderer::Sun(), states);
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
		//for (int x = 0; x < imgSize.x; x++)
		//{
		//	for (int y = 0; y < imgSize.y; y++)
		//	{
		//		glm::vec2 screenCoord(
		//			(2.0f * x) / imgSize.x - 1.0f,
		//			(-2.0f * y) / imgSize.y + 1.0f);
		//		Ray ray = info.camera.makeRay(screenCoord);
		//		poss.push_back(ray.origin);
		//		dirs.push_back(ray.direction);
		//	}
		//}
		float angle = glm::pi<float>() / 2;
		Ray ray = info.camera.makeRay({ 0, 0 });
		for (int i = 0; i < 1000; i++)
		{
			glm::vec3 dir = ray.direction;
			// generate random point on unit sphere
			float theta = Utils::get_random(0, glm::two_pi<float>()); // range 0 to 2pi
			float u = Utils::get_random(-1, 1); // range -1 to 1
			float squ = glm::sqrt(1 - u * u); // avoid computing this twice
			glm::vec3 offset;
			offset.x = glm::cos(theta) * squ;
			offset.y = glm::sin(theta) * squ;
			offset.z = u;

			// height of triangle, from tan(y/x)=angle
			// x = 1 since this is unit cone
			float radius = glm::atan(angle);
			dir += (offset * radius);
			glm::vec3 pos = ray.origin;
			poss.push_back(pos);
			dirs.push_back(glm::normalize(dir));
		}

		for (int i = 0; i < poss.size(); i++)
		{
			tClrs.push_back(glm::vec3(1));
			bClrs.push_back(glm::vec3(0));
		}

		lines = new LinePool(poss, dirs, tClrs, bClrs);
	}

	int& ShadowRays()
	{
		return info.numShadowRays;
	}
}
