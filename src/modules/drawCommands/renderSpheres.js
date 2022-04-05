
import {Geometry, Vector3, Matrix4} from "potree";
import {sphere} from "../geometries/sphere.js";


const shaderSource = `
struct Uniforms {
	worldView : mat4x4<f32>,
	proj : mat4x4<f32>,
	screen_width : f32,
	screen_height : f32,
};

struct Mat4s { values : array<mat4x4<f32>> };

@binding(0) @group(0) var<uniform> uniforms     : Uniforms;
@binding(1) @group(0) var<storage, read> worldViewArray : Mat4s;

struct VertexIn{
	@builtin(instance_index) instanceID    : u32,
	@location(0)             sphere_pos    : vec4<f32>,
	@location(1)             sphere_radius : f32,
	@location(2)             point_pos     : vec4<f32>,
	@location(3)             point_normal  : vec4<f32>,
};

struct VertexOut{
	@builtin(position)   out_pos   : vec4<f32>,
	@location(0)         fragColor : vec4<f32>,
};

struct FragmentIn{
	@location(0) fragColor : vec4<f32>,
};

struct FragmentOut{
	@location(0) outColor : vec4<f32>,
	@location(1) id : u32,
};

@stage(vertex)
fn main_vertex(vertex : VertexIn) -> VertexOut {

	var worldView = worldViewArray.values[vertex.instanceID];
	
	// var worldPos : vec4<f32> = vertex.sphere_pos + vertex.point_pos * vertex.sphere_radius;
	var worldPos : vec4<f32> = vertex.point_pos * vertex.sphere_radius;
	worldPos.w = 1.0;
	var viewPos : vec4<f32> = worldView * worldPos;

	var vout : VertexOut;
	vout.fragColor = vec4<f32>(vertex.point_normal.xyz, 1.0);
	vout.out_pos = uniforms.proj * viewPos;

	return vout;
}

@stage(fragment)
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.outColor = fragment.fragColor;
	fout.id = 0u;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let geometry_spheres = null;
let uniformBuffer = null;
let mat4Buffer;
let bindGroup = null;
let capacity = 10_000;
let f32Matrices = new Float32Array(16 * capacity);

function createPipeline(renderer){

	let {device} = renderer;

	let module = device.createShaderModule({code: shaderSource});

	pipeline = device.createRenderPipeline({
		vertex: {
			module: module,
			entryPoint: "main_vertex",
			buffers: [
				{ // sphere position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // sphere radius
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float32",
					}],
				},{ // sphere-vertices position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float32x3",
					}],
				},{ // sphere normal
					arrayStride: 4 * 3,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 3,
						offset: 0,
						format: "float32x3",
					}],
				}
			]
		},
		fragment: {
			module: module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	geometry_spheres = new Geometry({
		buffers: [{
			name: "position",
			buffer: new Float32Array(3 * capacity),
		},{
			name: "radius",
			buffer: new Float32Array(capacity),
		}]
	});

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 256;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		mat4Buffer = device.createBuffer({
			size: 64 * capacity,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: mat4Buffer}},
			],
		});
	}

	initialized = true;

}

function updateUniforms(drawstate){

	let {renderer, camera} = drawstate;

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	{ // transform
		let world = new Matrix4();
		let view = camera.view;
		let worldView = new Matrix4().multiplyMatrices(view, world);

		f32.set(worldView.elements, 0);
		f32.set(camera.proj.elements, 16);
	}

	{ // misc
		let size = renderer.getSize();

		view.setUint32(128, size.width, true);
		view.setUint32(132, size.height, true);
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(spheres, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let position = geometry_spheres.buffers.find(g => g.name === "position").buffer;
	let radius = geometry_spheres.buffers.find(g => g.name === "radius").buffer;
	let vboPosition = renderer.getGpuBuffer(position);
	let vboRadius = renderer.getGpuBuffer(radius);

	{

		let world = new Matrix4();
		let view = drawstate.camera.view;
		let worldView = new Matrix4();

		for(let i = 0; i < spheres.length; i++){
			let sphere = spheres[i];
			let pos = sphere[0];

			position[3 * i + 0] = pos.x;
			position[3 * i + 1] = pos.y;
			position[3 * i + 2] = pos.z;

			radius[i] = sphere[1];

			world.elements[12] = pos.x;
			world.elements[13] = pos.y;
			world.elements[14] = pos.z;
			
			worldView.multiplyMatrices(view, world);

			f32Matrices.set(worldView.elements, 16 * i);
		}

		let numSpheres = spheres.length;
		device.queue.writeBuffer(vboPosition, 0, position.buffer, 0, 12 * numSpheres);
		device.queue.writeBuffer(vboRadius, 0, radius.buffer, 0, 4 * numSpheres);
		device.queue.writeBuffer(mat4Buffer, 0, f32Matrices.buffer, 0, 64 * numSpheres);
	}

	{ // solid
		let sphereVertices = sphere.buffers.find(b => b.name === "position").buffer;
		let sphereNormals = sphere.buffers.find(b => b.name === "normal").buffer;
		let vboSphereVertices = renderer.getGpuBuffer(sphereVertices);
		let vboSphereNormals = renderer.getGpuBuffer(sphereNormals);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboRadius);
		passEncoder.setVertexBuffer(2, vboSphereVertices);
		passEncoder.setVertexBuffer(3, vboSphereNormals);

		let vboIndices = renderer.getGpuBuffer(sphere.indices);

		passEncoder.setIndexBuffer(vboIndices, "uint32");

		let numSpheres = spheres.length;
		let numVertices = sphere.numElements;
		// passEncoder.draw(numVertices, numSpheres, 0, 0);

		let numIndices = sphere.indices.length;
		passEncoder.drawIndexed(numIndices, numSpheres);
	}


};