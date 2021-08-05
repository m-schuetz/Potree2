
import {Vector3, Matrix4} from "potree";

const vs = `
[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	color : vec4<f32>;
};

struct Position{
	x : f32;
	y : f32;
	z : f32;
};

[[block]] struct U32s{
	values : [[stride(4)]] array<u32>;
};

[[block]] struct Positions{
	values : [[stride(12)]] array<Position>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<storage, read> indices : U32s;
[[binding(2), group(0)]] var<storage, read> positions : Positions;
[[binding(3), group(0)]] var<storage, read> triangleIDs : U32s;

struct VertexInput {
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOutput {
	[[builtin(position)]] position  : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	var abc = triangleIDs.values[0];

	var triangleIndex = vertex.vertexID / 3u;
	var vertexIndex = indices.values[vertex.vertexID];

	var p = positions.values[vertexIndex];
	var position = vec4<f32>(p.x, p.y, p.z, 1.0);

	// var triangleID = 1234u * triangleIndex; 
	// var triangleID = 1234u * triangleIDs.values[triangleIndex]; 
	var triangleID = triangleIDs.values[triangleIndex]; 

	// var inRange = triangleIndex > 403000u && triangleIndex < 404000u;
	// inRange = triangleIndex == 403000u;
	// if(!inRange){
	// 	triangleID = 0u;
	// }

	var R = (triangleID >>  0u) & 0xFFu;
	var G = (triangleID >>  8u) & 0xFFu;
	var B = (triangleID >> 16u) & 0xFFu;
	var r = f32(R) / 256.0;
	var g = f32(G) / 256.0;
	var b = f32(B) / 256.0;
	var color = vec4<f32>(r, g, b, 1.0);


	var output : VertexOutput;
	output.position = uniforms.proj * uniforms.worldView * position;
	output.color = color;

	return output;
}
`;

const fs = `

[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	color : vec4<f32>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;

struct FragmentInput {
	[[builtin(position)]] position  : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
	[[builtin(frag_depth)]] depth : f32;
};

[[stage(fragment)]]
fn main(fragment : FragmentInput) -> FragmentOutput {

	var output : FragmentOutput;

	output.color = fragment.color;

	output.depth = fragment.position.z;
	// output.depth = fragment.position.z * 1.001;

	return output;
}
`;

let initialized = false;
let pipelineCache = new Map();

function getPipeline(drawstate, node){

	let {renderer} = drawstate;
	let {device} = renderer;

	let material = node.material;
	let depthWrite = material.depthWrite;
	let id = `depthWrite(${depthWrite})`;

	if(pipelineCache.has(id)){
		return pipelineCache.get(id);
	}

	let pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [],
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: depthWrite,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	pipelineCache.set(id, pipeline);

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	initialized = true;
}

function updateUniforms(node, uniformsBuffer, drawstate){

	let {renderer, camera} = drawstate;
	let {device} = renderer;

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	{ // transform
		let world = node.world;
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		f32.set(worldView.elements, 0);
		f32.set(camera.proj.elements, 16);
	}

	{ // misc
		let material = node.material;

		view.setFloat32(128, material.color.x, true);
		view.setFloat32(132, material.color.y, true);
		view.setFloat32(136, material.color.z, true);
		view.setFloat32(140, 1.0, true);
	}

	device.queue.writeBuffer(uniformsBuffer, 0, data, 0, 256);
}

let uniformsMap = new Map();

function getUniformsBuffer(renderer, node){

	if(!uniformsMap.has(node)){
		const uniformBufferSize = 256; 

		let uniformsBuffer = renderer.device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		uniformsMap.set(node, uniformsBuffer);
	}

	return uniformsMap.get(node);
}

let bindGroupCache = new Map();
function getBindGroup(drawstate, node){

	let {renderer} = drawstate;
	let bindGroup = bindGroupCache.get(node);

	if(!bindGroup){

		let uniformBuffer = getUniformsBuffer(renderer, node);

		let pipeline = getPipeline(drawstate, node);

		let vboIndices = renderer.getGpuBuffer(node.geometry.indices);
		let vbos = renderer.getGpuBuffers(node.geometry);
		let vboPosition = vbos.find(item => item.name === "position").vbo;
		let vboTriangleIDs = vbos.find(item => item.name === "triangle_ids").vbo;
		
		bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: vboIndices}},
				{binding: 2, resource: {buffer: vboPosition}},
				{binding: 3, resource: {buffer: vboTriangleIDs}},
			]
		});

		bindGroupCache.set(node, bindGroup);
	}
	
	return bindGroup;
}

export function render(node, drawstate){
	
	let {renderer, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	let uniformsBuffer = getUniformsBuffer(renderer, node);
	updateUniforms(node, uniformsBuffer, drawstate);

	let pipeline = getPipeline(drawstate, node);
	let bindGroup = getBindGroup(drawstate, node);

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let numElements = node.geometry.indices.length;
	passEncoder.draw(numElements, 1, 0, 0);
}

export class TriangleColorMaterial{

	constructor(){
		this.color = new Vector3(1.0, 0.0, 0.5);
		this.uniformBufferData = new ArrayBuffer(256);
		this.depthWrite = true;
	}
	
}