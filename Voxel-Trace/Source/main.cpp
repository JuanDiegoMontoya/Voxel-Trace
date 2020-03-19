#include "stdafx.h"

#include "Engine.h"
#include "Renderer.h"
#include "Interface.h"
#include "Voxtrace.h"

int main()
{
	EngineConfig cfg;
	cfg.verticalSync = true;
	Engine::Init(cfg);
	Renderer::Init();
	Interface::Init();
	Voxels::Init();

	Engine::Run();

	Engine::Cleanup();

	return 0;
}