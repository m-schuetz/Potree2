
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, PotreeControls, StationaryControls, Mesh, RenderTarget} from "potree";
import {Renderer, Timer, EventDispatcher, InputHandler} from "potree";
import {drawTexture, loadImage, drawImage} from "./prototyping/textures.js";
import {geometries} from "potree";
import {Potree} from "potree";
import {loadGLB} from "potree";
import {MeasureTool} from "./interaction/measure.js";
import * as ProgressiveLoader from "./modules/progressive_loader/ProgressiveLoader.js";
import {readPixels, readDepth} from "./renderer/readPixels.js";
import {
	renderPoints, renderMeshes, renderQuads, 
	renderPointsOctree, renderQuadsOctree
} from "potree";
import {dilate, EDL, hqs_normalize} from "potree";
import Stats from "stats";
import * as TWEEN from "tween";

let frame = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;
let lastFrameTime = 0;

let renderer = null;
let camera = null;
let controls = null;
let measure = null;
let dbgImage = null;
let stats = null;
let inputHandler = null;

let dispatcher = new EventDispatcher();

let scene = new Scene();
let dbgSphere = null;

function addEventListener(name, callback){
	dispatcher.addEventListener(name, callback);
}

function removeEventListener(name, callback){
	dispatcher.removeEventListener(name, callback);
}

dispatcher.add("click", (e) => {
	console.log("click");
});

function initScene(){
	// {
	// 	let mesh = new Mesh("cube", geometries.cube);
	// 	mesh.scale.set(0.5, 0.5, 0.5);

	// 	scene.root.children.push(mesh);
	// }

	dbgSphere = new Mesh("sphere", geometries.sphere);
	dbgSphere.scale.set(0.1, 0.1, 0.1);
	// dbgSphere.renderLayer = 10;
	scene.root.children.push(dbgSphere);
}

function update(){
	let now = performance.now();

	if((now - lastFpsCount) >= 1000.0){

		fps = framesSinceLastCount;

		lastFpsCount = now;
		framesSinceLastCount = 0;
		Potree.state.fps = Math.floor(fps).toLocaleString();
	}

	let timeSinceLastFrame = (lastFrameTime - now) / 1000;
	frame++;
	framesSinceLastCount++;

	controls.update(timeSinceLastFrame);
	camera.world.copy(controls.world);

	camera.updateView();
	Potree.state.camPos = camera.getWorldPosition().toString(1);
	Potree.state.camTarget = controls.pivot.toString(1);
	Potree.state.camDir = camera.getWorldDirection().toString(1);

	let size = renderer.getSize();
	camera.aspect = size.width / size.height;
	camera.updateProj();

	dispatcher.dispatch("update");

	lastFrameTime = now;
}

let sumBuffer = null;
function getSumBuffer(renderer){

	if(sumBuffer){
		return sumBuffer;
	}

	let size = [128, 128, 1];
	let descriptor = {
		size: size,
		colorDescriptors: [
			{
				size: size,
				format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			},{
				size: size,
				format: "r32uint",
				usage: GPUTextureUsage.TEXTURE_BINDING 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
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

	sumBuffer = new RenderTarget(renderer, descriptor);

	return sumBuffer;

}

function startPass(renderer, target, args = {}){
	let view = target.colorAttachments[0].texture.createView();

	let colorAttachments = [{
		view, 
		loadOp: "clear", 
		clearValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 },
		storeOp: 'store',
	}];

	// let disable_multi_attachments = args.disable_multi_attachments ?? false;
	if(target.colorAttachments.length === 2){
		let view = target.colorAttachments[1].texture.createView();
		colorAttachments.push(
			{
				view, 
				loadOp: "clear", 
				clearValue: { r: 0, g: 0, b: 0, a: 0},
				storeOp: 'store',
			}
		);
	}

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadOp: "clear", depthClearValue: 0,
			depthStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	return {commandEncoder, passEncoder};
}

function revisitPass(renderer, target){
	let view = target.colorAttachments[0].texture.createView();

	let colorAttachments = [
		{view, loadOp: "load", storeOp: 'store'}
	];

	if(target.colorAttachments.length === 2){
		let view = target.colorAttachments[1].texture.createView();
		colorAttachments.push({view, loadOp: "load", storeOp: 'store'});
	}

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadOp: "load",
			depthStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	return {commandEncoder, passEncoder};
}

function startSumPass(renderer, target){
	let view = target.colorAttachments[0].texture.createView();

	let colorAttachments = [
		{
			view, 
			loadOp: "clear", 
			clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
			storeOp: 'store',
		}
	];

	// let disable_multi_attachments = args.disable_multi_attachments ?? false;
	if(target.colorAttachments.length === 2){
		let view = target.colorAttachments[1].texture.createView();
		colorAttachments.push(
			{
				view, 
				loadOp: "clear", 
				clearValue: { r: 0, g: 0, b: 0, a: 0},
				storeOp: 'store',
			}
		);
	}

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadOp: "load",
			depthStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	return {commandEncoder, passEncoder};
}

function endPass(pass){

	let {passEncoder, commandEncoder} = pass;

	passEncoder.end();
	let commandBuffer = commandEncoder.finish();
	renderer.device.queue.submit([commandBuffer]);
}

function renderBasic(){
	let layers = new Map();

	let stack = [scene.root];
	while(stack.length > 0){
		let node = stack.pop();

		if(!node.visible){
			continue;
		}

		let layer = layers.get(node.renderLayer);
		if(!layer){
			layer = {renderables: new Map()};
			layers.set(node.renderLayer, layer);
		}

		let renderables = layer.renderables;

		let nodeType = node.constructor.name;
		if(!renderables.has(nodeType)){
			renderables.set(nodeType, []);
		}
		renderables.get(nodeType).push(node);

		for(let child of node.children){

			child.updateWorld();
			child.world.multiplyMatrices(node.world, child.world);

			stack.push(child);
		}
	}

	let renderables = layers.get(0).renderables;

	renderer.start();
	// renderer.updateScreenbuffer();

	let screenbuffer = renderer.screenbuffer;

	let pass = startPass(renderer, screenbuffer);
	let drawstate = {renderer, camera, renderables, pass};

	for(let [key, nodes] of renderables){
		for(let node of nodes){
			if(typeof node.render !== "undefined"){
				node.render(drawstate);
			}
		}
	}

	renderer.renderDrawCommands(drawstate);

	endPass(pass);

	renderer.finish();
}

function renderNotSoBasic(){
	// Timer.setEnabled(true);

	Potree.state.renderedObjects = [];
	Potree.state.renderedElements = 0;


	let layers = new Map();

	let stack = [scene.root];
	while(stack.length > 0){
		let node = stack.pop();

		if(!node.visible){
			continue;
		}

		let layer = layers.get(node.renderLayer);
		if(!layer){
			layer = {renderables: new Map()};
			layers.set(node.renderLayer, layer);
		}

		let renderables = layer.renderables;

		let nodeType = node.constructor.name;
		if(!renderables.has(nodeType)){
			renderables.set(nodeType, []);
		}
		renderables.get(nodeType).push(node);

		for(let child of node.children){

			child.updateWorld();
			child.world.multiplyMatrices(node.world, child.world);

			stack.push(child);
		}
	}

	let renderables = layers.get(0).renderables;

	let points = renderables.get("Points") ?? [];
	let octrees = renderables.get("PointCloudOctree") ?? [];

	Potree.state.numPoints = 0;
	Potree.state.numNodes  = 0;

	for(let octree of octrees){
		octree.showBoundingBox = Potree.settings.showBoundingBox;
		octree.pointBudget = Potree.settings.pointBudget;
		octree.pointSize = Potree.settings.pointSize;

		if(Potree.settings.updateEnabled){
			octree.updateVisibility(camera, renderer);
		}

		let numPoints = octree.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
		let numNodes = octree.visibleNodes.length;

		Potree.state.numPoints += numPoints;
		Potree.state.numNodes  += numNodes;
	}

	Timer.frameStart(renderer);
	
	let hqsEnabled = Potree.settings.hqsEnabled;
	let edlEnabled = Potree.settings.edlEnabled;
	let dilateEnabled = Potree.settings.dilateEnabled;

	renderer.start();
	// renderer.updateScreenbuffer();

	let screenbuffer = renderer.screenbuffer;
	let fbo_source = screenbuffer;

	let fbo_0 = renderer.getFramebuffer("fbo intermediate 0");
	let fbo_1 = renderer.getFramebuffer("fbo intermediate 1");
	
	fbo_0.setSize(...screenbuffer.size);
	fbo_1.setSize(...screenbuffer.size);
	
	let forwardRendering = !(hqsEnabled || dilateEnabled || edlEnabled);

	let fboTarget = (!dilateEnabled && !edlEnabled) ? screenbuffer : fbo_0;
	
	if(hqsEnabled){

		Timer.timestampSep(renderer, "HQS(total)-start");

		let fbo_hqs_depth = renderer.getFramebuffer("hqs depth");
		let fbo_hqs_sum = getSumBuffer(renderer);

		fbo_hqs_sum.setSize(...screenbuffer.size);
		fbo_hqs_depth.setSize(...screenbuffer.size);

		{ // depth pass
			let pass = startPass(renderer, fbo_hqs_depth);
			let drawstate = {renderer, camera, renderables, pass};

			Timer.timestamp(pass.passEncoder, "HQS-depth-start");
			renderPointsOctree(octrees, drawstate, ["hqs-depth"]);
			Timer.timestamp(pass.passEncoder, "HQS-depth-end");

			endPass(pass);
		}

		{ // attribute pass
			fbo_hqs_sum.depth = fbo_hqs_depth.depth;

			let pass = startSumPass(renderer, fbo_hqs_sum);
			let drawstate = {renderer, camera, renderables, pass};

			Timer.timestamp(pass.passEncoder, "HQS-attributes-start");
			renderPointsOctree(octrees, drawstate, ["additive_blending"]);
			Timer.timestamp(pass.passEncoder, "HQS-attributes-end");

			endPass(pass);
		}

		{ // normalization pass
			let pass = startPass(renderer, fboTarget);
			let drawstate = {renderer, camera, renderables, pass};

			// Timer.timestamp(pass.passEncoder, "HQS-normalize-start");
			hqs_normalize(fbo_hqs_sum, drawstate);
			// Timer.timestamp(pass.passEncoder, "HQS-normalize-end");

			endPass(pass);
		}

		fbo_source = fboTarget;

		Timer.timestampSep(renderer, "HQS(total)-end");

	}else if(forwardRendering){

		// // render directly to screenbuffer
		// let pass = startPass(renderer, screenbuffer);
		// let drawstate = {renderer, camera, renderables, pass};

		// for(let [key, nodes] of renderables){
		// 	for(let node of nodes){
		// 		if(typeof node.render !== "undefined"){
		// 			node.render(drawstate);
		// 		}
		// 	}
		// }

		// renderer.renderDrawCommands(drawstate);

		// endPass(pass);
	}else{

		// render to intermediate framebuffer
		let pass = startPass(renderer, fbo_0);
		let drawstate = {renderer, camera, renderables, pass};

		renderPointsOctree(octrees, drawstate);

		endPass(pass);

		fbo_source = fbo_0;
	}


	// // DILATE
	// if(dilateEnabled && Potree.settings.pointSize >= 2){ // dilate
	// 	// let fboTarget = edlEnabled ? fbo_1 : screenbuffer;
	// 	let fboTarget = fbo_1;

	// 	let pass = startPass(renderer, fboTarget);
	// 	let drawstate = {renderer, camera, renderables, pass};

	// 	dilate(fbo_source, drawstate);

	// 	endPass(pass);

	// 	fbo_source = fboTarget;
	// }

	// renderer.drawBoundingBox(
	// 	new Vector3(0.0, 0.0, 0.0),
	// 	new Vector3(50, 50, 50),
	// 	new Vector3(0, 255, 0),
	// );
	// renderer.drawBoundingBox(
	// 	new Vector3(4323655, 511232, 4646856),
	// 	new Vector3(50, 50, 50),
	// 	new Vector3(0, 255, 0),
	// );

	{ // render everything but point clouds
		let pass = revisitPass(renderer, fbo_source);
		let drawstate = {renderer, camera, renderables, pass};

		for(let [key, nodes] of renderables){
			for(let node of nodes){
				let hasRender = typeof node.render !== "undefined";
				let isOctree = node.constructor.name === "PointCloudOctree";
				let isImages360 = node.constructor.name === "Images360";

				if(hasRender && !isOctree){
					node.render(drawstate);
				}
			}
		}

		renderer.renderDrawCommands(drawstate);

		endPass(pass);
	}

	// EDL
	if(edlEnabled){ 
		let pass = startPass(renderer, screenbuffer);
		let drawstate = {renderer, camera, renderables, pass};

		EDL(fbo_source, drawstate);

		endPass(pass);
	}

	{ // HANDLE PICKING

		let renderedObjects = Potree.state.renderedObjects;
		
		let mouse = inputHandler.mouse;
		let window = 3;
		let wh = window / 2;
		renderer.readPixels(fbo_source.colorAttachments[1].texture, mouse.x - wh, mouse.y - wh, window, window).then(buffer => {

			let maxID = Math.max(...new Uint32Array(buffer));

			if(maxID === 0){
				return;
			}

			let node = null;
			let counter = 0;
			for(let i = 0; i < renderedObjects.length; i++){
				let object = renderedObjects[i];

				if(maxID < counter + object.numElements){
					node = object.node;
					break;
				}

				counter += object.numElements;
			}

			let elementIndex = maxID - counter;

			if(node?.constructor.name === "PointCloudOctreeNode"){

				let pointBuffer = node.geometry.buffer;
				let view = new DataView(pointBuffer);

				let x = view.getFloat32(12 * elementIndex + 0, true);
				let y = view.getFloat32(12 * elementIndex + 4, true);
				let z = view.getFloat32(12 * elementIndex + 8, true);

				x = x + node.octree.position.x;
				y = y + node.octree.position.y;
				z = z + node.octree.position.z;

				let position = new Vector3(x, y, z);

				Potree.pickPosition.copy(position);

				Potree.hoveredItem = {
					type: node?.constructor.name + " (Point)",
					instance: node,
					node: node,
					pointIndex: elementIndex,
					position: position,
					object: node.octree,
				};

			}else if(node?.constructor.name === "Images360"){

				let images = node;
				let image = node.images[elementIndex];
				
				let position = image.position.clone().add(images.position);

				Potree.pickPosition.copy(position);

				node.setHovered(elementIndex);

				Potree.hoveredItem = {
					type: image?.constructor.name,
					image, images, position,
					object: images,
				};
			}else if(node?.constructor.name === "Mesh"){
				let {geometry} = node;
				let positions = geometry.buffers.find(buffer => buffer.name === "position");
				let view = new DataView(positions.buffer.buffer);

				let p0 = new Vector3(
					view.getFloat32(3 * 12 * elementIndex +  0, true),
					view.getFloat32(3 * 12 * elementIndex +  4, true),
					view.getFloat32(3 * 12 * elementIndex +  8, true),
				);
				let p1 = new Vector3(
					view.getFloat32(3 * 12 * elementIndex + 12, true),
					view.getFloat32(3 * 12 * elementIndex + 16, true),
					view.getFloat32(3 * 12 * elementIndex + 20, true),
				);
				let p2 = new Vector3(
					view.getFloat32(3 * 12 * elementIndex + 24, true),
					view.getFloat32(3 * 12 * elementIndex + 28, true),
					view.getFloat32(3 * 12 * elementIndex + 32, true),
				);

				let center = p0.clone().add(p1).add(p2).divideScalar(3);

				center.applyMatrix4(node.world);

				Potree.pickPosition.copy(center);

				Potree.hoveredItem = {
					type: node?.constructor.name,
					instance: node,
					node: node,
					pointIndex: elementIndex,
					position: center,
					object: node,
				};
				
				
			}else{
				Potree.hoveredItem = null;
			}

		});

		{
			let radius = controls.radius / 70;
			dbgSphere.position.copy(Potree.pickPosition);
			dbgSphere.scale.set(radius, radius, radius);
			dbgSphere.updateWorld();
		}

		for(let {x, y, callback} of Potree.pickQueue){
			let position = Potree.pickPosition;
			let distance = camera.getWorldPosition().distanceTo(position);
			callback({distance, position});
		}
		Potree.pickQueue.length = 0;

		if(Potree.hoveredItem){
			inputHandler.hoveredElements = [Potree.hoveredItem];
		}else{
			inputHandler.hoveredElements = [];
		}

		
	}


	renderer.finish();

	Timer.frameEnd(renderer);
}


function loop(time){

	stats.begin();

	TWEEN.update(time);

	update();
	renderNotSoBasic();

	stats.end();

	requestAnimationFrame(loop);
}

function dbgControls(){

	let str = `
	
		controls.set({
			yaw: ${controls.yaw},
			pitch: ${controls.pitch},
			radius: ${controls.radius},
			pivot:  [${controls.pivot.toArray().join(", ")}],
		});

	`;

	console.log(str);

}
window.dbgControls = dbgControls;

export async function init(){

	renderer = new Renderer();

	await renderer.init();

	let potree = {};

	camera = new Camera();
	controls = new OrbitControls(renderer.canvas);
	// controls = new PotreeControls(renderer.canvas);
	window.orbitControls = new OrbitControls(renderer.canvas);

	potree.controls = controls;
	potree.addEventListener = addEventListener;
	potree.removeEventListener = removeEventListener;
	potree.renderer = renderer;
	potree.scene = scene;
	potree.onUpdate = (callback) => {
		addEventListener("update", callback);
	};
	potree.setControls = (newControls) => {

		let oldControls = potree.controls;

		inputHandler.removeInputListener(controls.dispatcher);
		inputHandler.addInputListener(newControls.dispatcher);
		
		controls = newControls;
		potree.controls = controls;

		oldControls.dispatcher.dispatch("unfocused");
		newControls.dispatcher.dispatch("focused");
	};

	measure = new MeasureTool(potree);
	potree.measure = measure;

	inputHandler = new InputHandler(potree);
	potree.inputHandler = inputHandler;

	inputHandler.addInputListener(controls.dispatcher);
	inputHandler.addInputListener(measure.dispatcher);
	inputHandler.addInputListener(dispatcher);


	// make things available in dev tools for debugging
	window.camera = camera;
	window.controls = controls;
	window.scene = scene;
	window.renderer = renderer;

	stats = new Stats();
	stats.showPanel(0);
	// document.body.appendChild( stats.dom );

	initScene();
	Potree.scene = scene;

	// progressive loader
	let element = document.getElementById("canvas");
	ProgressiveLoader.install(element, {
		onSetup: (node) => {
			scene.root.children.push(node)
			console.log("setup done");
		},
		onProgress: (e) => {
			console.log("progress", e);
		}
	});

	requestAnimationFrame(loop);

	Potree.instance = potree;

	return potree;
}



