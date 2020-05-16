import {Matrix4} from "../math/Matrix4.js";
import {toWebgpuAttribute, webgpuToGlsl} from "../octree/PointAttributes.js";

let vs = `
#version 450

layout(set = 0, binding = 0) uniform Uniforms {
	mat4 worldViewProj;
	mat4 worldView;
	mat4 proj;
	ivec4 imin;
	vec4 offset;
	float screenWidth;
	float screenHeight;
	vec4 fScale;
	ivec4 iScale;
} uniforms;

// layout(location = 0) in ivec3 a_position;
// layout(location = 1) in ivec4 a_rgb;
// layout(location=2) in vec3 posBillboard;

<!-- POINT ATTRIBUTES -->

<!-- ACTIVE ATTRIBUTE -->


layout(location = 0) out vec4 vColor;

<!-- COLOR FUNCTION -->

void main() {

	vColor = vec4(getColor(), 1.0);


	ivec3 ipos = a_position / uniforms.iScale.xyz;

	// vec3 rescale = vec3(0.0, 0.0, 0.0);
	// if(ipos.x > 1000 * 1000){
	// 	rescale = 1000;
	// 	ipos.x = ipos.x / 1000;
	// }
	// if(ipos.y > 10000){
	// 	off.y = - 10000;
	// 	ipos.y = ipos.y - 10000;
	// }
	// if(ipos.y > 10000){
	// 	off.y = - 10000;
	// 	ipos.y = ipos.y - 10000;
	// }
	

	vec3 pos = vec3(ipos) * uniforms.fScale.xyz;

	pos = pos + uniforms.offset.xyz;

	gl_Position = uniforms.worldViewProj * vec4(pos, 1.0);

	float w = gl_Position.w;
	float pointSize = 5.0;
	gl_Position.x += w * pointSize * posBillboard.x / uniforms.screenWidth;
	gl_Position.y += w * pointSize * posBillboard.y / uniforms.screenHeight;

}
`;


let fs = `

#version 450

layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
	outColor = vColor;
	//outColor = vec4(vColor.xyz / 256.0, 1.0);
	// outColor = vec4(1.0, 0.0, 0.0, 1.0);
}

`;

let shader = null;

let f32_for_mat4 = new Float32Array(4 * 4);
let i32_vec4 = new Int32Array(4);

// see https://github.com/cx20/webgpu-test/blob/master/examples/webgpu/cube/index.js
function updateBufferData(device, dst, dstOffset, src, commandEncoder) {
	const [uploadBuffer, mapping] = device.createBufferMapped({
		size: src.byteLength,
		usage: GPUBufferUsage.COPY_SRC,
	});

	new src.constructor(mapping).set(src);
	uploadBuffer.unmap();

	commandEncoder = commandEncoder || device.createCommandEncoder();
	commandEncoder.copyBufferToBuffer(uploadBuffer, 0, dst, dstOffset, src.byteLength);

	return { commandEncoder, uploadBuffer };
}

function buildVertexShader(octree){
	let attributes = octree.attributes;

	let srcAttributes = [];

	let i = 0;
	for(let attribute of octree.attributes){

		let webgpu = toWebgpuAttribute(attribute);

		let glslType = webgpuToGlsl(webgpu.type);

		let glslAttributeName = "a_" + attribute.name.replace(/[^\w]/g, "_")

		let line = `layout(location = ${i}) in ${glslType} ${glslAttributeName};`;
		console.log(line);

		srcAttributes.push(line);
		i++;
	}

	srcAttributes.push(`layout(location = ${i}) in vec3 posBillboard;`);


	let activeAttribute = window.debug?.attribute ?? "rgb";
	let colorFunction = {
		"rgb": `
			vec3 getColor(){
				vec3 rgb = vec3(activeAttribute.xyz);

				if(length(rgb) > 2.0){
					rgb = rgb / 256.0;
				}
				if(length(rgb) > 2.0){
					rgb = rgb / 256.0;
				}

				return rgb;
			}`,
		"intensity": `
			vec3 getColor(){
				float w = float(activeAttribute) / 256.0;

				return vec3(w, w, w);
			}
		`,
	}[activeAttribute];

	




	let built = vs.replace("<!-- POINT ATTRIBUTES -->", srcAttributes.join("\n"));
	built = built.replace("<!-- ACTIVE ATTRIBUTE -->", `#define activeAttribute a_${activeAttribute}`);
	built = built.replace("<!-- COLOR FUNCTION -->", `${colorFunction}`);

	console.log(built);

	//let built = vs;

	return built;
}

let billboardBuffer = null;
function getBillboardBuffer(device){

	if(billboardBuffer === null){
		let values = [
			-1, -1, 0,
			1, -1, 0,
			1, 1, 0,
			-1, 1, 0
		];

		const [gpuBuffer, mapping] = device.createBufferMapped({
			size: values.length * 4,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
		});
		new Float32Array(mapping).set(values);
		gpuBuffer.unmap();

		billboardBuffer = gpuBuffer;

	}

	return billboardBuffer;
}

export function initializePointCloudOctreePipeline(octree){
	let {device} = this;

	let bindGroupLayout = device.createBindGroupLayout({
		entries: [{
			binding: 0,
			visibility: GPUShaderStage.VERTEX,
			type: "uniform-buffer"
		}]
	});

	let pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

	let vertexBuffers = [];
	for(let i = 0; i < octree.attributes.length; i++){
		let attribute = octree.attributes[i];

		let webgpu = toWebgpuAttribute(attribute);

		let value = {
			arrayStride: webgpu.byteSize,
			stepMode: "instance",
			attributes: [{
				shaderLocation: i,
				offset: 0,
				format: webgpu.type,
			}]
		};

		console.log(attribute.name, value);

		vertexBuffers.push(value);
	}

	// billboard position
	vertexBuffers.push({
		arrayStride: 4 * 4,
		attributes: [
			{ 
				shaderLocation: vertexBuffers.length,
				offset: 0,
				format: "float4"
			}
		]
	});

	let pipeline = device.createRenderPipeline({
		layout: pipelineLayout,
		vertexStage: {
			module: shader.vsModule,
			entryPoint: 'main'
		},
		fragmentStage: {
			module: shader.fsModule,
			entryPoint: 'main'
		},
		vertexState: { vertexBuffers: vertexBuffers },
		colorStates: [{
			format: this.swapChainFormat,
		}],
		primitiveTopology: 'triangle-strip',
		depthStencilState: {
			depthWriteEnabled: true,
			depthCompare: "less",
			format: "depth24plus-stencil8",
		}
	});

	return {
		pipeline: pipeline,
		bindGroupLayout: bindGroupLayout,
	};
}

export function initializePointCloudOctreeUniforms(octree, bindGroupLayout){
	let {device} = this;

	const uniformBufferSize = 3 * 64 + 16 + 16 + 16 + 16 + 16;

	let buffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	let bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [{
			binding: 0,
			resource: {
				buffer: buffer,
			},
		}],
	});

	let uniforms = {
		buffer: buffer,
		bindGroup: bindGroup,
		bindGroupLayout: bindGroupLayout,
	};

	return uniforms;
}

function getScaleComponents(scale){

	let iScale = new Int32Array([1, 1, 1]);
	let fScale = new Float32Array([0, 0, 0]);

	for(let i = 0; i < 3; i++){
		if(scale[i] < 0.0001){
			iScale[i] = 1000;
			fScale[i] = scale[i] * 1000;
		}else{
			fScale[i] = scale[i];
		}
	}
	
	return [iScale, fScale];
}

export function renderPointCloudOctree(octree, view, proj, state){

	let {device} = this;

	let activeAttribute = window.debug?.activeAttribute ?? "rgb";
	let shouldUpdateShader = activeAttribute !== shader?.activeAttribute;

	if(shader === null){

		shader = {
			activeAttribute: activeAttribute,
			vsModule: this.makeShaderModule('vertex', buildVertexShader(octree)),
			fsModule: this.makeShaderModule('fragment', fs),
		};

		return;
	}

	if(!octree.webgpu){
		let {pipeline, bindGroupLayout} = initializePointCloudOctreePipeline.bind(this)(octree);
		let uniforms = initializePointCloudOctreeUniforms.bind(this)(octree, bindGroupLayout);

		octree.webgpu = {
			pipeline: pipeline,
			bindGroupLayout: bindGroupLayout,
			uniforms: uniforms,
		};
	}

	let {webgpu} = octree;
	let {pipeline, uniforms} = webgpu;

	let transform = new Matrix4();
	let scale = new Matrix4();
	let translate = new Matrix4();
	let worldView = new Matrix4();
	let worldViewProj = new Matrix4();
	let identity = new Matrix4();

	let commandEncoder = device.createCommandEncoder();

	let passEncoder = commandEncoder.beginRenderPass(state.renderPassDescriptor);
	passEncoder.setPipeline(pipeline);

	for(let node of octree.visibleNodes){
		if(!node.webgpu){
			let buffers = this.initializeBuffers(node);

			node.webgpu = {
				buffers: buffers,
			};
		}

		let webgpuNode = node.webgpu;
		let {buffers} = webgpuNode;

		scale.makeScale(octree.scale.x, octree.scale.y, octree.scale.z);
		translate.makeTranslation(octree.position.x, octree.position.y, octree.position.z);
		transform.multiplyMatrices(translate, scale);
		worldView.multiplyMatrices(view, transform);
		worldViewProj.multiplyMatrices(proj, worldView);

		let [width, height] = [this.canvas.clientWidth, this.canvas.clientHeight];

		
		let offsets = new Float32Array(octree.loader.offset);
		let screenSize = new Float32Array([width, height]);
		let [iScale, fScale] = getScaleComponents(octree.loader.scale);
		//let fScale = new Float32Array(octree.loader.scale);

		f32_for_mat4.set(worldViewProj.elements)
		uniforms.buffer.setSubData(0, f32_for_mat4);
		f32_for_mat4.set(worldView.elements)
		uniforms.buffer.setSubData(64 + 0, f32_for_mat4);
		f32_for_mat4.set(proj.elements)
		uniforms.buffer.setSubData(128 + 0, f32_for_mat4);

		// uniforms.buffer.setSubData(0, new Float32Array(worldViewProj.elements));
		// uniforms.buffer.setSubData(64 + 0, new Float32Array(worldView.elements));
		// uniforms.buffer.setSubData(128 + 0, new Float32Array(proj.elements));

		i32_vec4.set([0, 0, 0, 0])
		uniforms.buffer.setSubData(128 + 64, i32_vec4);
		uniforms.buffer.setSubData(128 + 80, offsets);
		uniforms.buffer.setSubData(128 + 96, screenSize);
		uniforms.buffer.setSubData(128 + 112, fScale);
		uniforms.buffer.setSubData(128 + 128, iScale);

		// {
		// 	let U8 = Uint8Array;
		// 	let I32 = Int32Array;
		// 	let F32 = Float32Array;

		// 	let buffer = new ArrayBuffer(256 + 16);
		// 	let bufferU8 = new Uint8Array(buffer);
		// 	let view = new DataView(buffer);

		// 	// bufferU8.set(new U8(new F32(worldViewProj.elements)), 0);
		// 	// bufferU8.set(new U8(new F32(worldView.elements)), 64);
		// 	// bufferU8.set(new U8(new F32(proj.elements)), 128);
		// 	// bufferU8.set(new U8(new I32([0, 0, 0, 0])), 192);
		// 	// bufferU8.set(new U8(offsets), 208);
		// 	// bufferU8.set(new U8(screenSize), 224);
		// 	// bufferU8.set(new U8(fScale), 240);
		// 	// bufferU8.set(new U8(iScale), 256);

		// 	// uniforms.buffer.setSubData(0, bufferU8);
		// }


		

		

		// let { uploadBuffer: buffer1 } = updateBufferData(device, uniforms.buffer, 0, 
		// 	new Float32Array(worldViewProj.elements), 
		// 	commandEncoder);

		

		let i = 0;
		for(let buffer of buffers){
			passEncoder.setVertexBuffer(i, buffer.handle);

			i++;
		}
		passEncoder.setVertexBuffer(i, getBillboardBuffer(device));

		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(4, node.numPoints, 0, 0);



		// buffer1.destroy();
	}
		
	passEncoder.endPass();

	device.defaultQueue.submit([commandEncoder.finish()]);


}