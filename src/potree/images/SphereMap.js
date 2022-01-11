

import {SceneNode, Vector3, Matrix4, EventDispatcher} from "potree";

let shaderCode = `

[[block]] struct Uniforms {
	view             : mat4x4<f32>;
	proj             : mat4x4<f32>;
	screen_width     : f32;
	screen_height    : f32;
	fovy             : f32;
	fill             : f32;
	camdir           : vec3<f32>;

};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(5), group(0)]] var sphereSampler     : sampler;
[[binding(6), group(0)]] var sphereTexture     : texture_2d<f32>;

struct VertexIn{
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] uv : vec2<f32>;
	[[location(1)]] rayd : vec2<f32>;
	// [[location(0), interpolate(flat)]] pointID : u32;
};

struct FragmentIn{
	[[location(0)]] uv : vec2<f32>;
	[[location(1)]] rayd : vec2<f32>;
	// [[location(0), interpolate(flat)]] pointID : u32;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] point_id : u32;
};

fn rotate(x : f32, y : f32, angle : f32){

}

let PI = 3.141592653589793;

[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	_ = uniforms;

	var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
		vec3<f32>(-1.0, -1.0, 0.000001),
		vec3<f32>( 1.0, -1.0, 0.000001),
		vec3<f32>( 1.0,  1.0, 0.000001),

		vec3<f32>(-1.0, -1.0, 0.000001),
		vec3<f32>( 1.0,  1.0, 0.000001),
		vec3<f32>(-1.0,  1.0, 0.000001),
	);

	var pos = vec4<f32>(QUAD_POS[vertex.vertexID], 1.0);
	
	var fovy = uniforms.fovy;
	var aspect = uniforms.screen_width / uniforms.screen_height;
	var top = tan(fovy / 2.0);
	var bottom = -top;
	var right = top * aspect;
	var left = -right;

	var a_top = fovy * 0.5;
	var a_bottom = -fovy * 0.5;
	var a_left = -a_top * aspect;
	var a_right = a_top * aspect;

	var xy = uniforms.camdir.xy;
	var yaw = -atan2(uniforms.camdir.y, uniforms.camdir.x) - PI / 2.0;
	var pitch = atan2(uniforms.camdir.z, length(xy));

	// var rayd = vec2<f32>(uniforms.camdir.x, uniforms.camdir.y);
	// var rayd = vec2<f32>(pitch, 0.0);

	var rayd = vec2<f32>(0.0, 0.0);
	if(vertex.vertexID == 0u || vertex.vertexID == 3u){
		rayd = vec2<f32>(a_left + yaw, a_bottom + pitch);
	}elseif(vertex.vertexID == 1u){
		rayd = vec2<f32>(a_right + yaw, a_bottom + pitch);
	}elseif(vertex.vertexID == 2u || vertex.vertexID == 4u){
		rayd = vec2<f32>(a_right + yaw, a_top + pitch);
	}elseif(vertex.vertexID == 5u){
		rayd = vec2<f32>(a_left + yaw, a_top + pitch);
	}


	var vout : VertexOut;
	vout.uv = pos.xy;
	vout.position = pos;
	vout.rayd = rayd;

	return vout;
}

[[stage(fragment)]]
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;

	_ = sphereTexture;
	_ = sphereSampler;

	// fout.color = textureSample(sphereTexture, sphereSampler, fragment.uv);
	// fout.color = textureSample(sphereTexture, sphereSampler, fragment.rayd);
	// fout.color = vec4<f32>(fragment.uv, 0.0, 1.0);
	fout.color = vec4<f32>(
		0.0, //fragment.rayd.x, 
		fragment.rayd.y, 
		0.0, 1.0);

	var corrector = length(vec3<f32>(fragment.rayd, 1.0));

	var uv = vec2<f32>(
		fragment.rayd.x / (2.0 * PI),
		-(fragment.rayd.y + PI) / (2.0 * PI),
	);
	fout.color = textureSample(sphereTexture, sphereSampler, uv);


	return fout;
}

`;

let initialized = false;
let pipeline = null;
let sampler = null;

function init(renderer){

	if(initialized){
		return;
	}
	
	let {device} = renderer;

	let module = device.createShaderModule({code: shaderCode});

	pipeline = device.createRenderPipeline({
		vertex: {
			module,
			entryPoint: "main_vertex",
			buffers: [],
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
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: false,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	sampler = device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "repeat",
		addressModeV: "repeat",
		maxAnisotropy: 1,
	});

	initialized = true;
}

export class SphereMap extends SceneNode{

	constructor(){
		super(); 

		this.uniformBuffer = null;
		this.bindGroup = null;
		this.dispatcher = new EventDispatcher();
		this.image = null;
		this.textureNeedsUpdate = false;
		this.texture = null;

	}

	updateUniforms(drawstate){

		let {renderer, camera} = drawstate;
		let {device} = renderer;

		if(this.uniformBuffer === null){
			const uniformBufferSize = 256;

			this.uniformBuffer = device.createBuffer({
				size: uniformBufferSize,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});

			this.bindGroup = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{binding: 0,resource: {buffer: this.uniformBuffer}},
				],
			});
		}

		let data = new ArrayBuffer(256);
		let f32 = new Float32Array(data);
		let view = new DataView(data);

		{ // transform
			// let view = camera.view;
			// let rot = new Matrix4().copy(camera.world);
			// let campos = new Vector3().applyMatrix4(camera.world);
			// rot.translate(-campos.x, -campos.y, -campos.z);
			// rot.invert();


			f32.set(camera.view.elements, 0);
			f32.set(camera.proj.elements, 16);
		}

		{ // misc
			let size = renderer.getSize();

			let camdir = camera.getWorldDirection();

			view.setFloat32(128, size.width, true);
			view.setFloat32(132, size.height, true);
			view.setFloat32(136, Math.PI * camera.fov / 180, true);
			view.setFloat32(144, camdir.x, true);
			view.setFloat32(148, camdir.y, true);
			view.setFloat32(152, camdir.z, true);
			// view.setFloat32(136, 10.0, true);
			// view.setUint32(140, Potree.state.renderedElements, true);
			// view.setInt32(144, this.hoveredIndex ?? -1, true);
		}

		renderer.device.queue.writeBuffer(this.uniformBuffer, 0, data, 0, data.byteLength);
		
	}

	setImage(image){

		if(this.image === image){
			return;
		}

		this.image = image;
		this.textureNeedsUpdate = true;
	}

	updateTexture(drawstate){

		let renderer = drawstate.renderer;

		if(this.image === null && this.texture === null){
			// this.texture = renderer.createTexture(128, 128, {format: "rgba8unorm"});
		}else if(this.image !== null && this.textureNeedsUpdate){
			let texture = renderer.createTexture(this.image.width, this.image.height, {format: "rgba8unorm"});

			renderer.device.queue.copyExternalImageToTexture(
				{source: this.image}, 
				{texture: texture},
				[this.image.width, this.image.height, 1]
			);

			this.texture = texture;
			this.textureNeedsUpdate = false;
		}

	}

	render(drawstate){


		let {renderer} = drawstate;

		init(renderer);

		this.updateUniforms(drawstate);
		this.updateTexture(drawstate);

		if(!this.texture){
			return;
		}

		let {passEncoder} = drawstate.pass;

		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: this.uniformBuffer}},
				{binding: 5, resource: sampler},
				{binding: 6, resource: this.texture.createView()},
			]
		});

		passEncoder.setPipeline(pipeline);
		passEncoder.setBindGroup(0, bindGroup);

		passEncoder.draw(6, 1, 0, 0);

	}


}