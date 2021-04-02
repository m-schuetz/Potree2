
import {Vector3, Matrix4} from "../../math/math.js";

const vs = `
[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] numPointLights : u32;
	[[offset(132)]] color_source : u32;
	[[offset(144)]] color : vec4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

[[location(0)]] var<in> in_position : vec4<f32>;
[[location(1)]] var<in> in_normal : vec4<f32>;
[[location(2)]] var<in> in_uv : vec2<f32>;
[[location(3)]] var<in> in_color : vec4<f32>;

[[builtin(position)]] var<out> Position : vec4<f32>;

[[location(0)]] var<out> out_position : vec4<f32>;
[[location(1)]] var<out> out_normal : vec4<f32>;
[[location(2)]] var<out> out_uv : vec2<f32>;
[[location(3)]] var<out> out_color : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {

	Position = uniforms.proj * uniforms.worldView * in_position;

	out_uv = in_uv;
	out_position = uniforms.worldView * in_position;
	out_normal = in_normal;
	out_color = in_color;

	return;
}
`;

const fs = `

[[block]] struct PointLight {
	[[offset(0)]] position : vec4<f32>;
};

[[block]] struct PointLights {
	[[offset(0)]] values : [[stride(16)]] array<PointLight>;
};

[[block]] struct Uniforms {
	[[offset(0)]] worldView : mat4x4<f32>;
	[[offset(64)]] proj : mat4x4<f32>;
	[[offset(128)]] numPointLights : u32;
	[[offset(132)]] color_source : u32;
	[[offset(144)]] color : vec4<f32>;
};

[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<storage_buffer> pointLights : PointLights;
[[binding(2), group(0)]] var<uniform_constant> mySampler: sampler;
[[binding(3), group(0)]] var<uniform_constant> myTexture: texture_2d<f32>;

[[location(0)]] var<in> in_position : vec4<f32>;
[[location(1)]] var<in> in_normal : vec4<f32>;
[[location(2)]] var<in> in_uv : vec2<f32>;
[[location(3)]] var<in> in_color : vec4<f32>;

[[location(0)]] var<out> out_color : vec4<f32>;

fn getColor() -> vec4<f32>{

	var color : vec4<f32>;

	if(uniforms.color_source == 0){
		// VERTEX COLOR

		color = in_color;

	}elseif(uniforms.color_source == 1){
		// NORMALS

		color = vec4<f32>(0.0, 0.0, 1.0, 1.0);

	}elseif(uniforms.color_source == 2){
		// uniform color

		color = uniforms.color;

	}elseif(uniforms.color_source == 3){
		// TEXTURE

		color = textureSample(myTexture, mySampler, in_uv);

	}

	return color;
};

[[stage(fragment)]]
fn main() -> void {

	var light : vec3<f32>;
	
	for(var i : i32 = 0; i < uniforms.numPointLights; i = i + 1){

		var lightPos : vec4<f32> = pointLights.values[i].position;

		var L : vec3<f32> = normalize(lightPos.xyz - in_position.xyz);
		var V : vec3<f32> = vec3<f32>(0, 0, 1.0);
		var H : vec3<f32> = normalize(V + L);
		var N : vec3<f32> = (uniforms.worldView * vec4<f32>(in_normal.xyz, 0.0)).xyz;

		N = normalize(N);

		var lightColor : vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

		var diff : f32 = max(dot(N, L), 0.0);
		var diffuse : vec3<f32> = diff * lightColor;

		var shininess : f32 = 100.0;
		var spec : f32 = pow(max(dot(N, H), 0.0), shininess);
		var specular : vec3<f32> = lightColor * spec;
		specular = vec3<f32>(0.0, 0.0, 0.0);

		light.r = light.r + diffuse.r + specular.r;
		light.g = light.g + diffuse.g + specular.g;
		light.b = light.b + diffuse.b + specular.b;
	}

	light.r = 0.3 * light.r + 1.0;
	light.g = 0.3 * light.g + 1.0;
	light.b = 0.3 * light.b + 1.0;

	var color : vec4<f32> = getColor();

	out_color.r = color.r * light.r;
	out_color.g = color.g * light.g;
	out_color.b = color.b * light.b;
	out_color.a = 1.0;

	return;
}
`;


export let ColorMode = {
	VERTEX_COLOR: 0,
	NORMALS: 1,
	DIFFUSE_COLOR: 2,
	TEXTURE: 3
};

export class PhongMaterial{

	constructor(){
		this.image = null;
		this.colorMode = ColorMode.DIFFUSE_COLOR;
		this.color = new Vector3(1.0, 0.0, 0.5);
		this.uniformBufferData = new ArrayBuffer(256);
	}
	
}


let initialized = false;
let pipeline = null;
let ssbo_pointLights = null;
let sampler = null;
let uniformBufferCache = new Map();

function initialize(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		vertexStage: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
		},
		primitiveTopology: "triangle-list",
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		},
		vertexState: {
			vertexBuffers: [
				{ // position
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // normal
					arrayStride: 3 * 4,
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "float32x3",
					}],
				},{ // uv
					arrayStride: 2 * 4,
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float2",
					}],
				},{ // color
					arrayStride: 4,
					attributes: [{ 
						shaderLocation: 3,
						offset: 0,
						format: "unorm8x4",
					}],
				},
			],
		},
		rasterizationState: {
			cullMode: "none",
		},
		colorStates: [{
				format: "bgra8unorm",
		}],
	});

	let maxLights = 100;
	ssbo_pointLights = renderer.createBuffer(maxLights * 16);

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

function getUniformBuffer(renderer, node){

	let uniformBuffer = uniformBufferCache.get(node);

	if(!uniformBuffer){
		const uniformBufferSize = 256; 

		uniformBuffer = renderer.device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		uniformBufferCache.set(node, uniformBuffer);
	}

	return uniformBuffer;
}

let bindGroupCache = new Map();
function getBindGroup(renderer, node){
	let bindGroup = bindGroupCache.get(node);

	if(!bindGroup){

		let uniformBuffer = getUniformBuffer(renderer, node);
		let texture = renderer.getGpuTexture(node.material.image);

		bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: ssbo_pointLights}},
				{binding: 2, resource: sampler},
				{binding: 3, resource: texture.createView()},
			]
		});

		bindGroupCache.set(node, bindGroup);
	}
	
	return bindGroup;
}

export function render(renderer, pass, node, camera, renderables){
	
	let {device} = renderer;
	let {material} = node;

	initialize(renderer);

	let uniformBuffer = getUniformBuffer(renderer, node);
	let pointLights = renderables.get("PointLight") ?? [];

	{ // update uniforms
		let data = material.uniformBufferData
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
			view.setUint32(128, pointLights.length, true);
			view.setUint32(132, material.colorMode, true);
			view.setFloat32(144, material.color.x, true);
			view.setFloat32(148, material.color.y, true);
			view.setFloat32(152, material.color.z, true);
			view.setFloat32(156, 1.0);
		}

		device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
	}

	
	if(pointLights.length > 0){

		let data = new Float32Array(pointLights.length * 4);
		for(let i = 0; i < pointLights.length; i++){
			let light = pointLights[i];
			let lightPos = light.position.clone().applyMatrix4(camera.view);

			data[4 * i + 0] = lightPos.x;
			data[4 * i + 1] = lightPos.y;
			data[4 * i + 2] = lightPos.z;
			data[4 * i + 3] = 0.0;
		}
		
		device.queue.writeBuffer(
			ssbo_pointLights, 0,
			data.buffer, 0, data.byteLength
		);
	}


	let {passEncoder} = pass;
	let vbos = renderer.getGpuBuffers(node.geometry);

	passEncoder.setPipeline(pipeline);

	let bindGroup = getBindGroup(renderer, node);
	
	let bindGroupIndex = renderer.getNextBindGroup();
	passEncoder.setBindGroup(0, bindGroup);

	let vboPosition = vbos.find(item => item.name === "position").vbo;
	let vboNormal = vbos.find(item => item.name === "normal").vbo;
	let vboUV = vbos.find(item => item.name === "uv").vbo;
	let vboColor = vbos.find(item => item.name === "color").vbo;

	passEncoder.setVertexBuffer(0, vboPosition);
	passEncoder.setVertexBuffer(1, vboNormal);
	passEncoder.setVertexBuffer(2, vboUV);
	passEncoder.setVertexBuffer(3, vboColor);

	if(node.geometry.indices){
		let indexBuffer = renderer.getGpuBuffer(node.geometry.indices);

		passEncoder.setIndexBuffer(indexBuffer, "uint32", 0, indexBuffer.byteLength);

		let numIndices = node.geometry.indices.length;
		passEncoder.drawIndexed(numIndices);
	}else{
		let numElements = node.geometry.numElements;
		passEncoder.draw(numElements, 1, 0, 0);
	}

}

