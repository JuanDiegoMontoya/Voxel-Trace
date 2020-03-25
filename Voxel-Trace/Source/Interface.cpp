#include "stdafx.h"
#include "shader.h"
#include "camera.h"
#include "input.h"
#include "Pipeline.h"
#include "Engine.h"
#include "Renderer.h"
#include "Interface.h"

#include "Voxtrace.h"

namespace Interface
{
	namespace
	{
		bool activeCursor = true;
	}


	void Init()
	{
		Engine::PushRenderCallback(DrawImGui, 1);
		Engine::PushRenderCallback(Update, 2);
	}


	void Update()
	{
		if (Input::Keyboard().pressed[GLFW_KEY_GRAVE_ACCENT])
			activeCursor = !activeCursor;
		glfwSetInputMode(Engine::GetWindow(), GLFW_CURSOR, activeCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
	}


	void DrawImGui()
	{
		{
			ImGui::Begin("Shaders");
			if (ImGui::Button("Recompile Fullscreen Shader"))
			{
				delete Shader::shaders["fullscreen"];
				Shader::shaders["fullscreen"] = new Shader("fullscreen.vs", "fullscreen.fs");
			}
			ImGui::End();
		}

		{
			ImGui::Begin("Info");

			ImGui::Text("FPS: %.0f (%.1f ms)", 1.f / Engine::GetDT(), 1000.0 * Engine::GetDT());
			ImGui::NewLine();
			glm::vec3 pos = Renderer::GetPipeline()->GetCamera(0)->GetPos();
			if (ImGui::InputFloat3("Camera Position", &pos[0], 2))
				Renderer::GetPipeline()->GetCamera(0)->SetPos(pos);
			pos = Renderer::GetPipeline()->GetCamera(0)->front;
			ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
			glm::vec3 eu = Renderer::GetPipeline()->GetCamera(0)->GetEuler();
			ImGui::Text("Euler Angles: %.0f, %.0f, %.0f", eu.x, eu.y, eu.z);
			pos = pos * .5f + .5f;
			ImGui::SameLine();
			ImGui::ColorButton("visualization", ImVec4(pos.x, pos.y, pos.z, 1.f));

			ImGui::NewLine();
			ImGui::Text("Cursor: %s", !activeCursor ? "False" : "True");

			ImGui::End();
		}

		{
			ImGui::Begin("Ray Tracing");

			if (ImGui::Button("Camera Snapshot"))
				Voxels::CameraRaySnapshot();

			ImGui::SliderFloat3("Sun pos", &Renderer::Sun()->position[0], -30, 30);
			ImGui::SliderFloat("Sun radius", &Renderer::Sun()->radius, 0, 30);
			if (ImGui::SliderFloat("Sun angle", &Renderer::SunPos(), 0, 6.28f, "%.2f"))
			{

			}
			if (ImGui::SliderFloat("Sun dist", &Renderer::SunDist(), 0, 20.f, "%.1f"))
			{

			}

			ImGui::End();
		}
	}


	bool IsCursorActive()
	{
		return activeCursor;
	}
}