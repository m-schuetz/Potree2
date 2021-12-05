
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, PotreeControls, Mesh, RenderTarget} from "potree";
import {Renderer, Timer, EventDispatcher, InputHandler} from "potree";
import {drawTexture, loadImage, drawImage} from "./prototyping/textures.js";
import {geometries} from "potree";
import {Potree} from "potree";
// import {PointCloudOctree } from "potree";
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

	frame++;
	framesSinceLastCount++;

	controls.update();
	camera.world.copy(controls.world);

	camera.updateView();
	Potree.state.camPos = camera.getWorldPosition().toString(1);
	Potree.state.camTarget = controls.pivot.toString(1);
	Potree.state.camDir = camera.getWorldDirection().toString(1);

	let size = renderer.getSize();
	camera.aspect = size.width / size.height;
	camera.updateProj();

	dispatcher.dispatch("update");

}

let sumBuffer = null;
function getSumBuffer(renderer){

	if(sumBuffer){
		return sumBuffer;
	}

	let size = [128, 128, 1];
	let descriptor = {
		size: size,
		colorDescriptors: [{
			size: size,
			format: "rgba16float",
			usage: GPUTextureUsage.TEXTURE_BINDING 
				| GPUTextureUsage.RENDER_ATTACHMENT,
		}],
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

	let colorAttachments = [
		{view, loadValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }}
	];

	// let disable_multi_attachments = args.disable_multi_attachments ?? false;
	if(target.colorAttachments.length === 2){
		let view = target.colorAttachments[1].texture.createView();
		colorAttachments.push(
			{view, loadValue: { r: 0, g: 0, b: 0, a: 0}}
		);
	}

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadValue: 0,
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
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
		{view, loadValue: "load"}
	];

	if(target.colorAttachments.length === 2){
		let view = target.colorAttachments[1].texture.createView();
		colorAttachments.push({view, loadValue: "load"});
	}

	let renderPassDescriptor = {
		colorAttachments,
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadValue: "load",
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	return {commandEncoder, passEncoder};
}

function startSumPass(renderer, target){
	let view = target.colorAttachments[0].texture.createView();

	let renderPassDescriptor = {
		colorAttachments: [{
			view, 
			loadValue: { r: 0, g: 0, b: 0, a: 0.0 }
		}],
		depthStencilAttachment: {
			view: target.depth.texture.createView(),
			depthLoadValue: "load",
			depthStoreOp: "store",
			stencilLoadValue: 0,
			stencilStoreOp: "store",
		},
		sampleCount: 1,
	};

	const commandEncoder = renderer.device.createCommandEncoder();
	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

	return {commandEncoder, passEncoder};
}

function endPass(pass){

	let {passEncoder, commandEncoder} = pass;

	passEncoder.endPass();
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

	for(let octree of octrees){
		octree.showBoundingBox = Potree.settings.showBoundingBox;
		octree.pointBudget = Potree.settings.pointBudget;
		octree.pointSize = Potree.settings.pointSize;

		if(Potree.settings.updateEnabled){
			octree.updateVisibility(camera);
		}

		let numPoints = octree.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
		let numNodes = octree.visibleNodes.length;

		Potree.state.numPoints = numPoints;
		Potree.state.numNodes = numNodes;
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

			Timer.timestamp(pass.passEncoder, "HQS-normalize-start");
			hqs_normalize(fbo_hqs_sum, drawstate);
			Timer.timestamp(pass.passEncoder, "HQS-normalize-end");

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


	// DILATE
	if(dilateEnabled && Potree.settings.pointSize >= 2){ // dilate
		// let fboTarget = edlEnabled ? fbo_1 : screenbuffer;
		let fboTarget = fbo_1;

		let pass = startPass(renderer, fboTarget);
		let drawstate = {renderer, camera, renderables, pass};

		dilate(fbo_source, drawstate);

		endPass(pass);

		fbo_source = fboTarget;
	}

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

		// if(window.fbo){
		// 	let texture = window.fbo.colorAttachments[0].texture;
		// 	drawTexture(renderer, pass, texture, 0.1, 0.1, 0.2, 0.2);
		// }

		endPass(pass);
	}

	{ // HANDLE PICKING

		let renderedObjects = Potree.state.renderedObjects;
		
		let mouse = inputHandler.mouse;
		let window = 2;
		let wh = window / 2;
		renderer.readPixels(fbo_source.colorAttachments[1].texture, mouse.x - wh, mouse.y - wh, window, window).then(buffer => {

			let maxID = Math.max(...new Uint32Array(buffer));

			// let nodeCounter = (max >>> 20);
			// let pointIndex = max & 0b1111_11111111_11111111;

			// if(nodeCounter === 0 && pointIndex === 0){
			// 	return;
			// }

			// if(max === 1){
			// 	console.log("abc?");
			// }

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

				let pointBuffer = node.geometry.buffer.buffer;
				let view = new DataView(pointBuffer);

				let x = view.getFloat32(12 * elementIndex + 0, true);
				let y = view.getFloat32(12 * elementIndex + 4, true);
				let z = view.getFloat32(12 * elementIndex + 8, true);

				x = x + node.octree.position.x;
				y = y + node.octree.position.y;
				z = z + node.octree.position.z;

				let position = new Vector3(x, y, z);
				let distance = camera.getWorldPosition().distanceTo(position);
				let radius = distance / 50;

				dbgSphere.position.set(x, y, z);
				dbgSphere.scale.set(radius, radius, radius);
				dbgSphere.updateWorld();

				Potree.pickPosition.copy(position);
			}else if(node?.constructor.name === "Images360"){

				let image = node.images[elementIndex];
				
				let position = image.position;
				let distance = camera.getWorldPosition().distanceTo(position);
				let radius = distance / 50;

				dbgSphere.position.copy(position);
				dbgSphere.scale.set(radius, radius, radius);
				dbgSphere.updateWorld();

				Potree.pickPosition.copy(position);
			}
		});

		for(let {x, y, callback} of Potree.pickQueue){
			let position = Potree.pickPosition;
			let distance = camera.getWorldPosition().distanceTo(position);
			callback({distance, position});
		}
		Potree.pickQueue.length = 0;

		

	}

	// renderer.drawSphere(
	// 	// dbgSphere.position,
	// 	new Vector3(637290.76, 851209.905, 510.7),
	// 	200
	// );



	// {
	// 	let layer = layers.get(10);

	// 	if(layer){

	// 		let renderables = layer.renderables;

	// 		let pass = revisitPass(renderer, fbo_source);
	// 		let drawstate = {renderer, camera, renderables, pass};

	// 		let meshes = renderables.get("Mesh") ?? [];
	// 		renderMeshes(meshes, drawstate);

	// 		endPass(pass);
	// 	}

	// }

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
	// controls = new OrbitControls(renderer.canvas);
	controls = new PotreeControls(renderer.canvas);


	potree.controls = controls;
	potree.addEventListener = addEventListener;
	potree.removeEventListener = removeEventListener;
	potree.renderer = renderer;
	potree.scene = scene;
	potree.onUpdate = (callback) => {
		addEventListener("update", callback);
	};

	measure = new MeasureTool(potree);
	potree.measure = measure;

	inputHandler = new InputHandler(potree);
	potree.inputHandler = inputHandler;

	inputHandler.addInputListener(controls.dispatcher);
	inputHandler.addInputListener(measure.dispatcher);

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

	return potree;
}



