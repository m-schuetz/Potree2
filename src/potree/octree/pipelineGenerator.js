
import {Potree, SplatType} from "potree";

// let shaderModuleCache = new Map();

export async function makePipeline(renderer, args = {}){

	// console.log("generating octree pipeline");

	let {device} = renderer;
	let {octree, state, flags} = args;

	state.stage = "building";

	let shaderPath = `${import.meta.url}/../octree.wgsl`;
	let response = await fetch(shaderPath);
	let shaderSource = await response.text();

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

	let template_mapping_enum = "";
	let template_mapping_selection = "";
	let template_mapping_functions = "";

	// let mappings = [...octree.material.mappings].map(value => value[1]);
	let mappings = octree.material.mappings;
	mappings.forEach( (mapping, i) => {
		template_mapping_enum += `const MAPPING_${128 + i} = ${128 + i}u;\n`;
		template_mapping_selection += `
			if(attrib.mapping == MAPPING_${128 + i}){
				color = map_${128 + i}(pointID, attrib, node, position);
			}`;

		template_mapping_functions += mapping.wgsl.replaceAll(/fn .*\(/g, `fn map_${128 + i}(`);
	});

	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_ENUM>>", template_mapping_enum);
	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_SELECTION>>", template_mapping_selection);
	modifiedShaderSource = modifiedShaderSource.replace("<<TEMPLATE_MAPPING_FUNCTIONS>>", template_mapping_functions);
	
	console.groupCollapsed("compiling octree shader");
	console.log("==== SHADER ====");
	console.log(modifiedShaderSource);
	console.groupEnd();

	// use cache
	// let module;
	// if(shaderModuleCache.has(modifiedShaderSource)){
	// 	module = shaderModuleCache.get(modifiedShaderSource);
	// }else{
	// 	module = device.createShaderModule({code: modifiedShaderSource, label: "point cloud shader"});
	// 	shaderModuleCache.set(modifiedShaderSource, module);
	// 	console.log("create new module");
	// }

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
			},
			// {
			// 	binding: 1,
			// 	visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
			// 	buffer: {type: 'read-only-storage'},
			// }
		],
	});

	let splatType = SplatType.POINTS;

	if(args.flags.find(flag => flag === "SPLAT_TYPE_0")) {splatType = SplatType.POINTS;}
	if(args.flags.find(flag => flag === "SPLAT_TYPE_1")) {splatType = SplatType.QUADS;}
	if(args.flags.find(flag => flag === "SPLAT_TYPE_2")) {splatType = SplatType.VOXELS;}

	let topology = "point-list";

	if(splatType == SplatType.POINTS){
		topology = "point-list";
	}else if(splatType === SplatType.QUADS){
		topology = "triangle-list";
	}else if(splatType === SplatType.VOXELS){
		topology = "triangle-list";
	}

	const pipelinePromise = device.createRenderPipelineAsync({
		label: `pipeline ${args.key}`,
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
			topology: topology,
			cullMode: 'none',
			// cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: depthWrite,
			depthCompare: "greater-equal",
			format: "depth32float",
		},
	});

	// let tStart = Date.now();
	// console.log(`start: ${performance.now() / 1000}`);
	// pipelinePromise.then(pipeline => {
	// 	let duration = Date.now() - tStart;

	// 	console.log(`end: ${performance.now() / 1000}`);
	// 	console.log(`duration: ${duration}`);
	// });

	let nodesBindGroup = device.createBindGroup({
		layout: layout_3,
		entries: [
			{binding: 0, resource: {buffer: state.nodesGpuBuffer}},
			// {binding: 1, resource: {buffer: state.nodesGpuBuffer}},
		],
	});

	let uniformBindGroup = device.createBindGroup({
		layout: layout_0,
		entries: [
			{binding: 0, resource: {buffer: state.uniformBuffer}},
			{binding: 1, resource: {buffer: state.attributesDescGpuBuffer}},
			{binding: 2, resource: {buffer: state.colormapGpuBuffer}},
		],
	});

	// pipeline.dbg_topology = topology;

	state.pipelinePromise = pipelinePromise;
	state.uniformBindGroup = uniformBindGroup;
	state.nodesBindGroup = nodesBindGroup;
	state.shaderSource = shaderSource;
	state.splatType = splatType;
	state.stage = "ready";
}