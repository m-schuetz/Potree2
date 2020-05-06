
import {vs} from "./../shaders/pointcloud.vs.js";
import {fs} from "./../shaders/pointcloud.fs.js";

let shader = null;


let whitelist = ["position", "rgb"];


function buildVertexShader(octree){

	// let attributes = octree.attributes;

	// let srcAttributes = "";

	// let i = 0;
	// for(let attribute of octree.attributes){

	// 	if(!whitelist.includes(attribute.name)){
	// 		continue;
	// 	}

	// 	if(attribute.type === "double"){
	// 		continue;
	// 	}

	// 	let glslAttributeName = "a_" + attribute.name.replace(/[^\w]/g, "_")

	// 	let line = `layout(location = ${i}) in ivec4 ${glslAttributeName};\n`;

	// 	srcAttributes += line;
	// 	i++;
	// }

	// let built = vs.replace("<!-- POINT ATTRIBUTES -->", srcAttributes);

	// console.log(built);

	let built = vs;

	return built;
}



function getFormatname(attribute){
	let {type, numElements} = attribute;

	let formatname = "";

	// TODO

	if(attribute.type === "int32"){
		return "int3";
	}else if(attribute.type === "uint16"){
		return "uchar4";
	}
	

	return formatname;
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
	//for(let i = 0; i < octree.attributes.length; i++){
	let i = 0;
	for(let attribute of octree.attributes){

		if(!whitelist.includes(attribute.name)){
			continue;
		}

		if(attribute.type === "double"){
			continue;
		}

		let formatname = getFormatname(attribute);

		// TODO
		let stride = attribute.byteSize;
		if(stride === 6){
			stride = 4;
		}

		let value = {
			arrayStride: stride,
			attributes: [{
				shaderLocation: i,
				offset: 0,
				format: formatname,
			}]
		};

		console.log(value);

		vertexBuffers.push(value);
		i++;
	}
	// {
	// 	arrayStride: 3 * 4,
	// 	attributes: [
	// 		{ // position
	// 			shaderLocation: 0,
	// 			offset: 0,
	// 			format: "int3"
	// 		}
	// 	]
	// },

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
		vertexState: {
			vertexBuffers: vertexBuffers,
			//[
				// {
				// 	arrayStride: 3 * 4,
				// 	attributes: [
				// 		{ // position
				// 			shaderLocation: 0,
				// 			offset: 0,
				// 			format: "int3"
				// 		}
				// 	]
				// },{
				// 	arrayStride: 1 * 4,
				// 	attributes: [
				// 		{ // color
				// 			shaderLocation: 1,
				// 			offset: 0,
				// 			format: "uchar4"
				// 		}
				// 	]
				// }
			//]
		},
		colorStates: [
			{
				format: this.swapChainFormat,
				alphaBlend: {
					srcFactor: "src-alpha",
					dstFactor: "one-minus-src-alpha",
					operation: "add"
				}
			}
		],
		primitiveTopology: 'point-list',
		rasterizationState: {
			frontFace: "ccw",
			cullMode: 'none'
		},
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

	const uniformBufferSize = 4 * 16 + 3 * 4; 

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

export function renderPointCloudOctree(octree, view, proj, passEncoder){

	if(shader === null){
		shader = {
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

	let transform = mat4.create();
	let scale = mat4.create();
	let translate = mat4.create();
	let worldView = mat4.create();
	let worldViewProj = mat4.create();
	let identity = mat4.create();

	for(let node of octree.visibleNodes){
		if(!node.webgpu){
			let buffers = this.initializeBuffers(node);

			node.webgpu = {
				buffers: buffers,
			};
		}

		let webgpuNode = node.webgpu;
		let {buffers} = webgpuNode;

		mat4.scale(scale, identity, octree.scale.toArray());
		mat4.translate(translate, identity, octree.position.toArray());
		mat4.multiply(transform, translate, scale);

		mat4.multiply(worldView, view, transform);
		mat4.multiply(worldViewProj, proj, worldView);

		uniforms.buffer.setSubData(0, worldViewProj);
		// uniforms.buffer.setSubData(4 * 16, new Int32Array([41650162, 55830631, 225668106]));
		uniforms.buffer.setSubData(4 * 16, new Int32Array([0, 0, 0]));

		passEncoder.setPipeline(pipeline);

		let bufPos = buffers.find(b => b.name === "position");
		let bufCol = buffers.find(b => b.name === "rgb");
		passEncoder.setVertexBuffer(0, bufPos.handle);
		passEncoder.setVertexBuffer(1, bufCol.handle);

		// let i = 0;
		// for(let attribute of octree.attributes){
			
		// 	if(!whitelist.includes(attribute.name)){
		// 		continue;
		// 	}

		// 	let buffer = buffers[i];

		// 	passEncoder.setVertexBuffer(i, buffer.handle);
		// 	i++;
		// }
		
		passEncoder.setBindGroup(0, uniforms.bindGroup);

		passEncoder.draw(node.numPoints, 1, 0, 0);

	}

}