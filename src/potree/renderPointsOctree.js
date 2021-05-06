
import {Vector3, Matrix4} from "potree";
import {Timer} from "potree";
import {generate as generatePipeline} from "./octree/pipelineGenerator.js";

let octreeStates = new Map();

function getOctreeState(renderer, octree, attributeName){

	let {device} = renderer;

	let attributes = octree.loader.attributes.attributes;
	let attribute = attributes.find(a => a.name === attributeName);

	let mapping = attributeName === "rgba" ? "rgba" : "scalar";

	let key = `${attribute.name}_${attribute.numElements}_${attribute.type.name}_${mapping}}`;

	let state = octreeStates.get(key);

	if(!state){
		let pipeline = generatePipeline(renderer, {attribute, mapping});

		const uniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		const uniformBindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
			],
		});

		state = {
			pipeline: pipeline,
			uniformBuffer: uniformBuffer,
			uniformBindGroup: uniformBindGroup,
		};

		octreeStates.set(key, state);
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
	
	let attributeName = Potree.settings.attribute;

	let octreeState = getOctreeState(renderer, octree, attributeName);

	updateUniforms(octree, octreeState, drawstate);

	let {pipeline, uniformBindGroup} = octreeState;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);

	let nodes = octree.visibleNodes;
	let i = 0;
	for(let node of nodes){

		let vboPosition = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "position").buffer);
		// let vboColor = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "rgba").buffer);
		// let vboIntensity = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === "intensity").buffer);
		let vboAttribute = renderer.getGpuBuffer(node.geometry.buffers.find(s => s.name === attributeName).buffer);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboAttribute);
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