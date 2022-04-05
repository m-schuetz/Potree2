

import {SceneNode, Vector3, Matrix4, EventDispatcher, geometries} from "potree";

let shaderCode = `

struct Uniforms {
	worldView        : mat4x4<f32>;
	proj             : mat4x4<f32>;
	screen_width     : f32;
	screen_height    : f32;
	fovy             : f32;
	fill             : f32;
	camdir           : vec3<f32>;

};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
[[binding(5), group(0)]] var sphereSampler     : sampler;
[[binding(6), group(0)]] var sphereTexture     : texture_2d<f32>;

struct VertexIn{
	@builtin(vertex_index) vertexID : u32,
	@location(0) position    : vec4<f32>,
	@location(1) uv          : vec2<f32>,
	@location(2) normal      : vec4<f32>,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
	@location(1) rayd : vec2<f32>,
	// [[location(0), interpolate(flat)]] pointID : u32,
};

struct FragmentIn{
	@location(0) uv : vec2<f32>,
	@location(1) rayd : vec2<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
	@location(1) point_id : u32,
};

fn rotate(x : f32, y : f32, angle : f32){

}

let PI = 3.141592653589793;

@stage(vertex)
fn main_vertex(vertex : VertexIn) -> VertexOut {

	_ = uniforms;

	var vout : VertexOut;
	vout.uv = vertex.uv;
	vout.position = uniforms.proj * uniforms.worldView * vec4<f32>(100000.0 * vertex.position.xyz, 1.0);
	// vout.rayd = rayd;


	// var QUAD_POS : array<vec3<f32>, 6> = array<vec3<f32>, 6>(
	// 	vec3<f32>(-1.0, -1.0, 0.000001),
	// 	vec3<f32>( 1.0, -1.0, 0.000001),
	// 	vec3<f32>( 1.0,  1.0, 0.000001),

	// 	vec3<f32>(-1.0, -1.0, 0.000001),
	// 	vec3<f32>( 1.0,  1.0, 0.000001),
	// 	vec3<f32>(-1.0,  1.0, 0.000001),
	// );

	// var pos = vec4<f32>(QUAD_POS[vertex.vertexID], 1.0);
	
	// var fovy = uniforms.fovy;
	// var aspect = uniforms.screen_width / uniforms.screen_height;
	// var top = tan(fovy / 2.0);
	// var bottom = -top;
	// var right = top * aspect;
	// var left = -right;

	// var a_top = fovy * 0.5;
	// var a_bottom = -fovy * 0.5;
	// var a_left = -a_top * aspect;
	// var a_right = a_top * aspect;

	// var xy = uniforms.camdir.xy;
	// var yaw = -atan2(uniforms.camdir.y, uniforms.camdir.x) - PI / 2.0;
	// var pitch = atan2(uniforms.camdir.z, length(xy));

	// // var rayd = vec2<f32>(uniforms.camdir.x, uniforms.camdir.y);
	// // var rayd = vec2<f32>(pitch, 0.0);

	// var rayd = vec2<f32>(0.0, 0.0);
	// if(vertex.vertexID == 0u || vertex.vertexID == 3u){
	// 	rayd = vec2<f32>(a_left + yaw, a_bottom + pitch);
	// }else if(vertex.vertexID == 1u){
	// 	rayd = vec2<f32>(a_right + yaw, a_bottom + pitch);
	// }else if(vertex.vertexID == 2u || vertex.vertexID == 4u){
	// 	rayd = vec2<f32>(a_right + yaw, a_top + pitch);
	// }else if(vertex.vertexID == 5u){
	// 	rayd = vec2<f32>(a_left + yaw, a_top + pitch);
	// }


	// var vout : VertexOut;
	// vout.uv = pos.xy;
	// vout.position = pos;
	// vout.rayd = rayd;

	return vout;
}

@stage(fragment)
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	_ = sphereTexture;
	_ = sphereSampler;

	var fout : FragmentOut;

	// var uv = vec2<f32>(
	// 	-fragment.uv.x - 51.0 / 128.0,
	// 	fragment.uv.y
	// );
	var uv = vec2<f32>(
		-fragment.uv.x,
		fragment.uv.y
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
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // uv
					arrayStride: 2 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float32x2",
					}],
				},{ // normal
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float32x3",
					}],
				}
			],
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

			// let angle = image.rotation.
			// let yaw = new Matrix4().rotate(angle, new Vector3(0, 0, 1));

			let imageRotation = this.imageRotation ?? new Vector3();
			let [a, b, c, d, e, f] = window.factors ?? [0, 0, 0, 0, 0, 0];

			let yaw = new Matrix4().rotate(
				a * imageRotation.x + b,
				new Vector3(0, 0, 1));
			let pitch = new Matrix4().rotate(
				c * imageRotation.y + d,
				new Vector3(1, 0, 0));
			let roll = new Matrix4().rotate(
				e * imageRotation.z + f,
				new Vector3(0, 1, 0));
			let rotate = new Matrix4().multiply(roll).multiply(pitch).multiply(yaw);

			let toCamera = new Matrix4().translate(...camera.getWorldPosition().toArray());
			// let rotate = this.rotation;
			let world = new Matrix4().multiplyMatrices(toCamera, rotate);
			// let world = toCamera;
			let view = camera.view;
			let worldView = new Matrix4().multiplyMatrices(view, world);


			f32.set(worldView.elements, 0);
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

		let sphere = geometries.sphere;

		let spherePositions = sphere.buffers.find(b => b.name === "position").buffer;
		let sphereUVs = sphere.buffers.find(b => b.name === "uv").buffer;
		let sphereNormals = sphere.buffers.find(b => b.name === "normal").buffer;
		let vboPositions = renderer.getGpuBuffer(spherePositions);
		let vboUVs = renderer.getGpuBuffer(sphereUVs);
		let vboNormals = renderer.getGpuBuffer(sphereNormals);

		passEncoder.setVertexBuffer(0, vboPositions);
		passEncoder.setVertexBuffer(1, vboUVs);
		passEncoder.setVertexBuffer(2, vboNormals);

		// passEncoder.draw(6, 1, 0, 0);

		let indexBuffer = renderer.getGpuBuffer(sphere.indices);

		passEncoder.setIndexBuffer(indexBuffer, "uint32", 0, indexBuffer.byteLength);

		let numIndices = sphere.indices.length;
		passEncoder.drawIndexed(numIndices);

	}


}