
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";


const vs = `
[[block]] struct Uniforms {
	[[size(64)]] worldView : mat4x4<f32>;
	[[size(64)]] proj : mat4x4<f32>;
	[[size(4)]] screen_width : f32;
	[[size(4)]] screen_height : f32;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

struct VertexInput {
	[[builtin(instance_index)]] instanceIdx : u32;
	[[builtin(vertex_index)]] vertexID : u32;
	[[location(0)]] position : vec4<f32>;
	[[location(1)]] color : vec4<f32>;
	// [[location(2)]] intensity : u32;
};

struct VertexOutput {
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};


[[stage(vertex)]]
fn main(vertex : VertexInput) -> VertexOutput {

	var output : VertexOutput;

	var viewPos : vec4<f32> = uniforms.worldView * vertex.position;
	output.position = uniforms.proj * viewPos;

	var c : vec4<f32> = vertex.color;
	// var c : vec4<f32> = vec4<f32>(
	// 	f32(vertex.intensity) / 256.0 + vertex.color.x * 0.001, 
	// 	f32(vertex.intensity) / 256.0, 
	// 	f32(vertex.intensity) / 256.0, 
	// 	1.0
	// );

	// check if instance_index works
	// c.r = f32(vertex.instanceIdx);
	// c.g = 0.0;
	// c.b = 0.0;
	// c.a = 1.0;

	output.color = c;

	return output;
}
`;

const fs = `

struct FragmentInput {
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOutput {
	[[location(0)]] color : vec4<f32>;
};

[[stage(fragment)]]
fn main(fragment : FragmentInput) -> FragmentOutput {
	var output : FragmentOutput;
	output.color = fragment.color;

	return output;
}
`;

let octreeStates = new Map();

function createPipeline(renderer){

	let {device} = renderer;

	const pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // point position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "unorm8x4",
					}],
				},
				// { // intensity
				// 	arrayStride: 4,
				// 	stepMode: "vertex",
				// 	attributes: [{ 
				// 		shaderLocation: 2,
				// 		offset: 0,
				// 		format: "uint32",
				// 	}],
				// },
			],
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	return pipeline;
}

function getOctreeState(renderer, node){

	let {device} = renderer;

	let state = octreeStates.get(node);

	if(!state){
		let pipeline = createPipeline(renderer);

		const uniformBuffer = device.createBuffer({
			size: 2 * 4 * 16 + 8,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uNodesBuffer = device.createBuffer({
			size: 16 * 1000,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				// {binding: 1, resource: {buffer: uNodesBuffer}}
			],
		});

		state = {
			pipeline: pipeline,
			uniformBuffer: uniformBuffer,
			uniformBindGroup: uniformBindGroup,
			uNodesBuffer: uNodesBuffer,
		};

		octreeStates.set(node, state);

	}

	return state;
}

function updateUniforms(octree, octreeState, drawstate){

	let {uniformBuffer} = octreeState;
	let {renderer} = drawstate;

	let data = new ArrayBuffer(256);
	let f32 = new Float32Array(data);
	let view = new DataView(data);

	let world = octree.world;
	let camView = camera.view;
	let worldView = new Matrix4().multiplyMatrices(camView, world);

	f32.set(worldView.elements, 0);
	f32.set(camera.proj.elements, 16);

	let size = renderer.getSize();

	view.setFloat32(128, size.width, true);
	view.setFloat32(132, size.height, true);

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, 136);
}

function renderOctree(octree, drawstate, passEncoder){
	
	let {renderer} = drawstate;
	
	let octreeState = getOctreeState(renderer, octree);

	updateUniforms(octree, octreeState, drawstate);

	let {pipeline, uniformBindGroup} = octreeState;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	let nodes = octree.visibleNodes;
	let i = 0;
	for(let node of nodes){

		let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);
		let vboColor = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "rgba").buffer);
		// let vboIntensity = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "intensity").buffer);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboColor);
		// passEncoder.setVertexBuffer(2, vboIntensity);

		if(octree.showBoundingBox === true){
			let box = node.boundingBox.clone().applyMatrix4(octree.world);
			let position = box.min.clone();
			position.add(box.max).multiplyScalar(0.5);
			// position.applyMatrix4(octree.world);
			let size = box.size();
			// let color = new Vector3(...SPECTRAL.get(node.level / 5));
			let color = new Vector3(255, 255, 0);
			renderer.drawBoundingBox(position, size, color);
		}

		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, i);

		i++;
	}
}

export function render(args = {}){

	let octrees = args.in;
	let target = args.target;
	let drawstate = args.drawstate;
	let {renderer, camera} = drawstate;

	let firstDraw = target.version < renderer.frameCounter;
	let view = target.colorAttachments[0].texture.createView();
	let loadValue = firstDraw ? { r: 0.1, g: 0.2, b: 0.3, a: 1.0 } : "load";
	let depthLoadValue = firstDraw ? 0 : "load";

	// loadValue = "load";
	// depthLoadValue = "load";

	let renderPassDescriptor = {
		colorAttachments: [{view, loadValue}],
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadValue: depthLoadValue,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};
	target.version = renderer.frameCounter;

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	Timer.timestamp(passEncoder, "octree-start");

	for(let octree of octrees){
		renderOctree(octree, drawstate, passEncoder);
	}

	Timer.timestamp(passEncoder, "octree-end");

	passEncoder.endPass();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);

};