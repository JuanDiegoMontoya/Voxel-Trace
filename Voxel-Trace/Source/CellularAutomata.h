#pragma once
#include "cuda_runtime.h"

struct Cell
{
#ifdef N__CUDACC__
	template<int X, int Y, int Z>
	virtual void Update(glm::ivec3 pos, Cell* grid, glm::ivec3 bound) {}
#endif
};

/*
	There is an idea of a Cellular Automata, some kind 
	of abstraction, but there is no real CA, only an 
	entity, something illusory. Although CA can hide 
	its cold gaze, and you can shake its hand and feel 
	flesh gripping yours, and even sense our lifestyles 
	are probably comparable, CA is simply not there.
*/
template<typename C, int X, int Y, int Z>
class CellularAutomata
{
public:
	//CellularAutomata(int X, int Y, int Z) : numBlocks((X* Y* Z + blockSize - 1) / blockSize) {}
	CellularAutomata()
	{
		cudaMallocManaged(&this->Grid, X * Y * Z * sizeof(C));
		cudaMallocManaged(&this->TGrid, X * Y * Z * sizeof(C));
	}
	~CellularAutomata()
	{
		cudaFree(Grid);
		cudaFree(TGrid);
	}

	virtual void Init() {}
	virtual void Update() {}
	virtual void Render() {}

protected:
	const int blockSize = 256;
	const int numBlocks = (X * Y * Z + blockSize - 1) / blockSize;

	bool UpdateMesh = false;
	virtual void genMesh() {}

	class Mesh* mesh_ = nullptr;
	C* Grid  = nullptr;
	C* TGrid = nullptr;
};

// excuse this jank but it works as an "interface
typedef CellularAutomata<Cell, 0, 0, 0> CAInterface;