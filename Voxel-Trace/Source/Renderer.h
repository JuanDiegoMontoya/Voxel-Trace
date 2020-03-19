#pragma once
#include "CellularAutomata.h"

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
	void initCubeMap();
	void drawCubeMap();

	// debug
	void drawAxisIndicators();

	// CA stuff
	Pipeline* GetPipeline();
	CAInterface* GetAutomaton();
	void SetAutomatonIndex(int index);
	int GetAutomatonIndex();
	float& GetUpdateFrequencyRef();
}