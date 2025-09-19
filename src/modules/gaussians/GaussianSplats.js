
import {SceneNode, Vector3, Vector4, Matrix4, Box3, Frustum, EventDispatcher, StationaryControls, RenderTarget} from "potree";
import {Timer} from "potree";
import {compose} from "./compose.js";
import {RadixSortKernel} from "radix-sort-esm";

let initializing = false;
let initialized = false;
let pipeline = null;
let uniformsBuffer = new ArrayBuffer(512);
let uniformsGpuBuffer = null;
let layout = null;
let fbo_blending = null;

let splatSortKeys = null;
let splatSortValues = null;
let pipeline_depth = null;

let dbg = 0;

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

	if(initialized) return;
	if(initializing) return;

	initializing = true;
	
	let {device} = renderer;

	uniformsGpuBuffer = renderer.createBuffer({
		size: uniformsBuffer.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let size = [128, 128, 1];
	let descriptor = {
		size: size,
		colorDescriptors: [
			{
				size: size,
				format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			}
		],
		depthDescriptor: {
			size: size,
			format: "depth32float",
			usage: GPUTextureUsage.TEXTURE_BINDING 
				| GPUTextureUsage.RENDER_ATTACHMENT,
		}
	};

	fbo_blending = new RenderTarget(renderer, descriptor);

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
			},{
				binding: 5,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	let shaderPath = `${import.meta.url}/../gaussians.wgsl`;
	let response = await fetch(shaderPath);
	let shaderSource = await response.text();

	let module = device.createShaderModule({code: shaderSource});

	let tStart = Date.now();

	let blend = {
		color: {
			// srcFactor: "one",
			// dstFactor: "one-minus-src-alpha",
			srcFactor: "one-minus-dst-alpha",
			dstFactor: "one",
			operation: "add",
		},
		alpha: {
			// srcFactor: "one",
			// dstFactor: "one-minus-src-alpha",
			srcFactor: "one-minus-dst-alpha",
			dstFactor: "one",
			operation: "add",
		},
	};

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
				{format: "rgba16float", blend: blend}
			],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: false,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});
	let duration = Date.now() - tStart;


	{ // sort stuff
		splatSortKeys        = renderer.createBuffer({size: 4 * 10_000_000});
		splatSortValues      = renderer.createBuffer({size: 4 * 10_000_000});

		let shaderPath = `${import.meta.url}/../gaussians_distance.wgsl`;
		let response = await fetch(shaderPath);
		let shaderSource = await response.text();

		let module = device.createShaderModule({code: shaderSource});
		pipeline_depth = device.createComputePipeline({
			layout: "auto",
			compute: {module: module}
		});
	}


	initialized = true;
}

export class GaussianSplats extends SceneNode{

	constructor(url){
		super(); 

		this.url = url;
		this.dispatcher = new EventDispatcher();
		this.initialized = false;

		this.positions = new Float32Array([
			0.2, 0.2, 0.0,
			0.4, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.2, 0.0,
			0.4, 0.4, 0.0,
			0.2, 0.4, 0.0,
		]);

		this.splatData = null;
		this.numSplatsUploaded = 0;
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
			view.setUint32 (offset + 20, this.numSplats, true);
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

		fbo_blending.setSize(...renderer.screenbuffer.size);

		let colorAttachments = [{
			view: fbo_blending.colorAttachments[0].texture.createView(), 
			loadOp: "clear", 
			clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
			storeOp: 'store',
		}];

		let renderPassDescriptor = {
			colorAttachments,
			depthStencilAttachment: {
				view: renderer.screenbuffer.depth.texture.createView(),
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
			sampleCount: 1,
		};

		if(this.numSplats === 0) return;

		if(!this.radixSortKernel || this.radixSortKernel.count != this.numSplats){
			this.radixSortKernel = new RadixSortKernel({
				device,
				keys: splatSortKeys,
				values: splatSortValues,
				count: this.numSplats,
				bit_count: 32,
			})
		}

		// Transfer data to GPU
		if(this.splatData && splatBuffers.numSplats === 0){
			// create splat buffer
			splatBuffers.numSplats = this.numSplats;
			splatBuffers.position = renderer.createBuffer({size: this.numSplats * 12});
			splatBuffers.color    = renderer.createBuffer({size: this.numSplats * 16});
			splatBuffers.rotation = renderer.createBuffer({size: this.numSplats * 16});
			splatBuffers.scale    = renderer.createBuffer({size: this.numSplats * 12});
		}

		if(this.numSplatsLoaded > this.numSplatsUploaded){
			let transfer = (target, source, offset, size) => {
				device.queue.writeBuffer(
					target, offset,
					source, offset, 
					size 
				);
			};

			let numNew = this.numSplatsLoaded - this.numSplatsUploaded;
			transfer(splatBuffers.position, this.splatData.positions, 12 * this.numSplatsUploaded, 12 * numNew);
			transfer(splatBuffers.color   , this.splatData.color    , 16 * this.numSplatsUploaded, 16 * numNew);
			transfer(splatBuffers.rotation, this.splatData.rotation , 16 * this.numSplatsUploaded, 16 * numNew);
			transfer(splatBuffers.scale   , this.splatData.scale    , 12 * this.numSplatsUploaded, 12 * numNew);

			this.numSplatsUploaded += numNew;

			// transfer(splatBuffers.position, this.splatData.positions);
			// transfer(splatBuffers.color, this.splatData.color);
			// transfer(splatBuffers.rotation, this.splatData.rotation);
			// transfer(splatBuffers.scale, this.splatData.scale);
		}

		const commandEncoder = renderer.device.createCommandEncoder();

		{ // SORT
			let pass = commandEncoder.beginComputePass()

			// First, create a buffer of depth values
			let bindGroup = device.createBindGroup({
				layout: pipeline_depth.getBindGroupLayout(0),
				entries: [
				{ binding: 0, resource: { buffer: uniformsGpuBuffer }},
				{ binding: 1, resource: { buffer: splatBuffers.position }},
				{ binding: 2, resource: { buffer: splatSortKeys }},
				{ binding: 3, resource: { buffer: splatSortValues }},
				],
			});

			pass.setPipeline(pipeline_depth);
			pass.setBindGroup(0, bindGroup);
			let numGroups = Math.ceil(Math.sqrt(this.numSplats / 256));
			pass.dispatchWorkgroups(numGroups, numGroups, 1);

			// then sort
			this.radixSortKernel.dispatch(pass);
			pass.end();


		}

		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
		Timer.timestamp(passEncoder, "gaussians-start");

		this.updateUniforms(drawstate);

		// let {passEncoder} = drawstate.pass;
		passEncoder.setPipeline(pipeline);

		// Bind groups should be cached...but I honestly don't care. 
		// Why you can't just pass pointers to resources in WebGPU, like in modern 
		// bindless APIs and even OpenGL since 2010 with NV_shader_buffer_load, remains a mystery.
		let bindGroup = device.createBindGroup({
			layout: layout,
			entries: [
				{binding: 0, resource: {buffer: uniformsGpuBuffer}},
				{binding: 1, resource: {buffer: splatSortValues}},
				{binding: 2, resource: {buffer: splatBuffers.position}},
				{binding: 3, resource: {buffer: splatBuffers.color}},
				{binding: 4, resource: {buffer: splatBuffers.rotation}},
				{binding: 5, resource: {buffer: splatBuffers.scale}},
			],
		});

		passEncoder.setBindGroup(0, bindGroup);
		passEncoder.draw(6 * this.numSplats);
		passEncoder.end();
		
		Timer.timestamp(passEncoder, "gaussians-end");

		let commandBuffer = commandEncoder.finish();
		renderer.device.queue.submit([commandBuffer]);
		
		compose(renderer, 
			fbo_blending, 
			renderer.screenbuffer
		);
	}

}