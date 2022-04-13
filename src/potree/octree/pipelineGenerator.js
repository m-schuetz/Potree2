
import {generate as generateShaders} from "./shaderGenerator.js";
import {Potree} from "potree";
import { Attribute_Custom } from "../PointCloudMaterial.js";

export async function* generate(renderer, args = {}){

	console.log("generating octree pipeline");

	let {device} = renderer;
	let {octree, flags} = args;

	let shaderPath = `${import.meta.url}/../octree.wgsl`;
	let response = await fetch(shaderPath);
	let shaderSource = await response.text();

	yield "source loaded";

	let depthWrite = true;
	let blend = {
		color: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
		alpha: {
			srcFactor: "one",
			dstFactor: "zero",
			operation: "add",
		},
	};

	let isAdditive = flags.includes("additive_blending");
	let format = "bgra8unorm";

	// isAdditive = true;
	if(isAdditive){
		format = "rgba16float";
		depthWrite = false;

		blend = {
			color: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
			alpha: {
				srcFactor: "one",
				dstFactor: "one",
				operation: "add",
			},
		};
	}

	let modifiedShaderSource = shaderSource;
	// let customAttributes = [];
	// for(let [name, attribute] of octree.material.attributes){
	// 	if(attribute instanceof Attribute_Custom){
	// 		// modifiedShaderSource += attribute.wgsl;
	// 		customAttributes.push([name, attribute]);
	// 	}
	// }

	let template_mapping_enum = "";
	let template_mapping_selection = "";
	let template_mapping_functions = "";

	let mappings = [...octree.material.mappings].map(value => value[1]);
	mappings.forEach( (mapping, i) => {
		template_mapping_enum += `let MAPPING_${128 + i} = ${128 + i}u;`;
		template_mapping_selection += `
			else if(attribute.mapping == MAPPING_${128 + i}){
				color = map_${128 + i}(vertex, attribute, node, position);
			}`;

		template_mapping_functions += mapping.wgsl.replaceAll(/fn .*\(/g, `fn map_${128 + i}(`);
	});

	console.log(mappings);


	// for(let i = 0; i < mappings.length; i++){
	// 	let attribute = customAttributes[i][1];
	// 	template_mapping_enum += `let MAPPING_${128 + i} = ${128 + i}u;`;
	// 	template_mapping_selection += `
	// 		else if(attribute.mapping == MAPPING_${128 + i}){
	// 			color = map_${128 + i}(vertex, attribute, node, position);
	// 		}`;

	// 	template_mapping_functions += attribute.wgsl.replaceAll(/fn .*\(/g, `fn map_${128 + i}(`);
	// }

	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_ENUM>>", template_mapping_enum);
	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_SELECTION>>", template_mapping_selection);
	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_FUNCTIONS>>", template_mapping_functions);
	
	console.groupCollapsed("compiling octree shader");
	console.log("==== SHADER ====");
	console.log(modifiedShaderSource);
	console.groupEnd();

	let module = device.createShaderModule({code: modifiedShaderSource, label: "point cloud shader"});

	const layout_0 = renderer.device.createBindGroupLayout({
		label: "point cloud shader bind group layout",
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
			}
		],
	});

	const layout_1 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				sampler: {sampleType: 'filtering'},
			},{
				binding: 1,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				sampler: {sampleType: 'filtering'},
			},{
				binding: 2,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				texture: {sampleType : 'float'},
			}
		],
	});

	const layout_2 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	const layout_3 = renderer.device.createBindGroupLayout({
		entries: [
			{
				binding: 0,
				visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
				buffer: {type: 'read-only-storage'},
			}
		],
	});

	const pipeline = device.createRenderPipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [
				layout_0,
				layout_1,
				layout_2,
				layout_3,
			],
		}),
		vertex: {
			module: module,
			entryPoint: "main_vertex",
			buffers: [],
		},
		fragment: {
			module: module,
			entryPoint: "main_fragment",
			targets: [
				{format: format, blend: blend},
				{format: "r32uint", blend: undefined}
			],
		},
		primitive: {
			topology: 'point-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: depthWrite,
			depthCompare: "greater",
			format: "depth32float",
		},
	});

	return {pipeline, shaderSource, stage: "created pipeline"};
}