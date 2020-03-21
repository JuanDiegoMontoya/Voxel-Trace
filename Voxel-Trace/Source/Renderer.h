#pragma once

// responsible for making stuff appear on the screen
class Pipeline;
namespace Renderer
{
	void Init();
	void CompileShaders();

	// interaction
	void Update();
	void DrawAll();
	void Clear();

	void drawQuad();

	// debug
	void drawAxisIndicators();

	// CA stuff
	Pipeline* GetPipeline();
}