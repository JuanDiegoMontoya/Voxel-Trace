#pragma once
#include "CellularAutomata.h"

//https://tutcris.tut.fi/portal/files/4312220/kellomaki_1354.pdf p74
struct WaterCell : public Cell
{
	float depth = 0; // d += -dt*(SUM(Q)/(dx)^2) # sum of all adjacent pipes
};

struct Pipe
{
	float flow = 0; // Q += A*(g/dx)*dh*dt
};

struct PipeUpdateArgs
{
	float g = 9.8;
	float dx = 1; // length of pipe
	float dt = .125;
};

struct SplashArgs
{
	glm::vec2 pos = glm::vec2(40, 80);
	float A = 30; // amplitude
	float b = .1; // falloff rate
};

template<int X, int Y, int Z>
class PipeWater : public CellularAutomata<WaterCell, X, Y, Z>
{
public:
	PipeWater();
	~PipeWater();
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

private:
	virtual void genMesh() override;

	std::vector<glm::vec3> vertices; // order doesn't change
	std::vector<glm::vec2> vertices2d; // order doesn't change
	std::vector<GLuint> indices; // immutable basically
	class IBO* pIbo = nullptr;
	class VBO* pVbo = nullptr;
	class VAO* pVao = nullptr;

	// use 
	void initDepthTex();
	GLuint HeightTex;
	struct cudaGraphicsResource* imageResource;
	struct cudaArray* arr;

	const int PBlockSize = 128;
	const int hPNumBlocks = ((X+1) * Z + PBlockSize - 1) / PBlockSize;
	const int vPNumBlocks = (X * (Z+1) + PBlockSize - 1) / PBlockSize;
	Pipe* hPGrid = nullptr; // horizontal (x axis)
	Pipe* vPGrid = nullptr; // vertical (z axis)
	Pipe* temphPGrid = nullptr; // temp
	Pipe* tempvPGrid = nullptr; // temp

	PipeUpdateArgs args;
	bool calcNormals = true;
	glm::ivec2 splashLoc;

	SplashArgs splash;
};