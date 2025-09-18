
import {SceneNode, Vector3, Vector4, Matrix4, Box3, Frustum, EventDispatcher, StationaryControls} from "potree";

let initialized = false;
let pipeline = null;
let uniformsBuffer = new ArrayBuffer(512);
let uniformsGpuBuffer = null;
let layout = null;
let bindGroup = null;
let stateCache = new Map();

let defaultSampler = null;

let splatBuffers = {
	numSplats: 0,

	position:  null,
	color:     null,
	rotation:  null,
	scale:     null,
};

// some reusable variables to reduce GC strain
let _fm        = new Matrix4();
let _frustum   = new Frustum();
let _world     = new Matrix4();
let _worldView = new Matrix4();
let _rot       = new Matrix4();
let _trans     = new Matrix4();
let _pos       = new Vector4();
let _pos2      = new Vector4();
let _box       = new Box3();
let _dirx      = new Vector3();
let _diry      = new Vector3();
let _dirz      = new Vector3();

async function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	uniformsGpuBuffer = renderer.createBuffer({
		size: uniformsBuffer.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// splatBuffer = renderer.createBuffer({size: 128});

	layout = renderer.device.createBindGroupLayout({
		label: "gaussian splat uniforms",
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'uniform'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			},{
				binding: 2,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			},{
				binding: 3,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			},{
				binding: 4,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	// bindGroup = device.createBindGroup({
	// 	layout: layout,
	// 	entries: [
	// 		{binding: 0, resource: {buffer: uniformsGpuBuffer}},
	// 		{binding: 1, resource: {buffer: splatBuffer}},
	// 	],
	// });

	let shaderPath = `${import.meta.url}/../gaussians.wgsl`;
	let response = await fetch(shaderPath);
	let shaderSource = await response.text();

	let module = device.createShaderModule({code: shaderSource});

	let tStart = Date.now();

	pipeline = device.createRenderPipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [layout],
		}),
		vertex: {
			module,
			entryPoint: "main_vertex",
			buffers: []
		},
		fragment: {
			module,
			entryPoint: "main_fragment",
			targets: [
				{format: "bgra8unorm"},
				{format: "r32uint"},
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});
	let duration = Date.now() - tStart;

	initialized = true;
}

export class GaussianSplats extends SceneNode{

	constructor(url){
		super(); 

		this.url = url;
		this.dispatcher = new EventDispatcher();

		this.positions = new Float32Array([
			0.2, 0.2, 0.0,
			0.4, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.4, 0.0,
		]);

		this.splatData = null;
	}

	setHovered(index){
		// this.hoveredIndex = index;
		// this.dispatcher.dispatch("hover", {
		// 	images: this,
		// 	index: index,
		// 	image: this.images[index],
		// });
	}

	updateUniforms(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		let f32 = new Float32Array(uniformsBuffer);
		let view = new DataView(uniformsBuffer);

		{ // transform
			this.updateWorld();
			let world = this.world;
			let view = camera.view;
			_worldView.multiplyMatrices(view, world);

			f32.set(_worldView.elements, 0);
			f32.set(world.elements, 16);
			f32.set(view.elements, 32);
			f32.set(camera.proj.elements, 48);
			// debugger;
		}

		{ // misc
			let size = renderer.getSize();

			let offset = 256;

			view.setFloat32(offset + 0, size.width, true);
			view.setFloat32(offset + 4, size.height, true);
			view.setFloat32(offset + 8, 10.0, true);
			view.setUint32 (offset + 12, Potree.state.renderedElements, true);
			view.setInt32  (offset + 16, this.hoveredIndex ?? -1, true);
		}

		renderer.device.queue.writeBuffer(uniformsGpuBuffer, 0, uniformsBuffer, 0, uniformsBuffer.byteLength);
	}

	project(coord){

		if(this.projector){
			return this.projector.forward(coord);
		}else{
			return coord;
		}

	}

	render(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		init(renderer);
		if(!initialized) return;

		if(this.splatData && splatBuffers.numSplats === 0){
			// create splat buffer
			splatBuffers.numSplats = this.numSplats;
			splatBuffers.position = renderer.createBuffer({size: this.numSplats * 12});
			splatBuffers.color    = renderer.createBuffer({size: this.numSplats * 16});
			splatBuffers.rotation = renderer.createBuffer({size: this.numSplats * 16});
			splatBuffers.scale    = renderer.createBuffer({size: this.numSplats * 12});
			
			let transfer = (target, source) => {
				device.queue.writeBuffer(
					target, 0,
					source, 0, source.byteLength 
				);
			};

			transfer(splatBuffers.position, this.splatData.positions);
			transfer(splatBuffers.color, this.splatData.color);
			transfer(splatBuffers.rotation, this.splatData.rotation);
			transfer(splatBuffers.scale, this.splatData.scale);
		}

		if(splatBuffers.numSplats === 0) return;

		// TODO: Sort splats

		init(renderer);

		if(!initialized) return;

		this.updateUniforms(drawstate);

		let {passEncoder} = drawstate.pass;
		passEncoder.setPipeline(pipeline);

		// Bind groups should be cached...but I honestly don't care. 
		// Why you can't just pass pointers to resources in WebGPU, like in modern 
		// bindless APIs and even OpenGL since 2010 with NV_shader_buffer_load, remains a mystery.
		let bindGroup = device.createBindGroup({
			layout: layout,
			entries: [
				{binding: 0, resource: {buffer: uniformsGpuBuffer}},
				{binding: 1, resource: {buffer: splatBuffers.position}},
				{binding: 2, resource: {buffer: splatBuffers.color}},
				{binding: 3, resource: {buffer: splatBuffers.rotation}},
				{binding: 4, resource: {buffer: splatBuffers.scale}},
			],
		});
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(6 * this.numSplats);
	}


}