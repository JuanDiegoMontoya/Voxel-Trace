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

namespace Voxels
{
	namespace
	{
		PerspectiveRayCamera cam;
		LinePool* lines = nullptr;

		// world description
		Block* blocks = nullptr;
		glm::vec3 chunkDim = { 10, 10, 10 };
		int numBlocks = chunkDim.x * chunkDim.y * chunkDim.z;

		// screen info
		glm::vec2 screenDim = { 100, 200 };
		
		// rendering shiz
		VBO* vbo = nullptr;
		VAO* vao = nullptr;
		GLuint screenTexture = -1;

		// cuda GL stuff
		cudaGraphicsResource* imageResource;
		cudaArray* arr;
	}

	void Init()
	{
		Engine::PushRenderCallback(Render, 5);
		InitGLStuff();
	}

	void InitGLStuff()
	{
		// TODO: move this to Vertices.h or something
		std::vector<glm::vec2> screenTexCoords =
		{
			{-1,-1 }, { 0, 0 },
			{ 1,-1 }, { 1, 0 },
			{ 1, 1 }, { 1, 1 },
			{-1,-1 }, { 0, 0 },
			{ 1, 1 }, { 1, 1 },
			{-1, 1 }, { 0, 1 },
		};

		// setup screen texture pointers
		vbo = new VBO(&screenTexCoords[0], 
			screenTexCoords.size() * sizeof(glm::vec2), GL_STATIC_DRAW);
		VBOlayout layout;
		layout.Push<float>(2); // pos
		layout.Push<float>(2); // texcoord
		vao = new VAO();
		vao->AddBuffer(*vbo, layout);

		// generate screen texture memory
		glGenTextures(1, &screenTexture);
		glBindTexture(GL_TEXTURE_2D, screenTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
			screenDim.x, screenDim.y, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);

		// init texture color
		glm::vec4 defColor(0, 1, 1, 1); // cyan
		glClearTexImage(screenTexture, 0, GL_RGBA, GL_FLOAT, &defColor[0]);

		// register the texture as a cuda resource
		cudaCheck(cudaGraphicsGLRegisterImage(&imageResource, screenTexture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
	}

	void Render()
	{
		auto c = Renderer::GetPipeline()->GetCamera(0);
		cam = PerspectiveRayCamera(c->GetPos(), c->GetPos() + c->GetDir(), 
			glm::vec3(0, 1, 0), glm::radians(30.f), 1920.f / 1080.f);
		
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

		// draw fullscreen quad
		ShaderPtr s = Shader::shaders["fullscreen"];
		s->Use();
		s->setInt("tex", 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, screenTexture);
		vao->Bind();
		glDrawArrays(GL_TRIANGLES, 0, 6);
		vao->Unbind();
		s->Unuse();
	}

	void CameraRaySnapshot()
	{
		delete lines; // ok if null (e.g. first instance)

		std::vector<glm::vec3> poss, dirs, tClrs, bClrs;

		glm::vec2 imgSize(19, 11);
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


surface<void, 2> screenSurface;