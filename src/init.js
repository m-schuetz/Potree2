
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, Mesh, RenderTarget} from "potree";
import {Renderer, Timer, EventDispatcher} from "potree";
import {drawTexture, loadImage, drawImage} from "./prototyping/textures.js";
import {geometries} from "potree";
import {Potree} from "potree";
import {loadGLB} from "potree";
import {MeasureTool} from "./interaction/measure.js";
import * as ProgressiveLoader from "./modules/progressive_loader/ProgressiveLoader.js";
import {readPixels, readDepth} from "./renderer/readPixels.js";
import {renderPoints, renderMeshes, renderPointsCompute, renderPointsOctree, renderPointsOctreeBundledVBO} from "potree";
import {dilate, EDL, hqs_normalize} from "potree";
import Stats from "stats";

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

let dispatcher = new EventDispatcher();

let scene = new Scene();

function addEventListener(name, callback){
	dispatcher.addEventListener(name, callback);
}

function removeEventListener(name, callback){
	dispatcher.removeEventListener(name, callback);
}

function initScene(){
	{
		let mesh = new Mesh("cube", geometries.cube);
		mesh.scale.set(0.5, 0.5, 0.5);

		scene.root.children.push(mesh);
	}
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
			format: "rgba32float",
			usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.RENDER_ATTACHMENT,
		}],
		depthDescriptor: {
			size: size,
			format: "depth32float",
			usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.RENDER_ATTACHMENT,
		}
	};

	sumBuffer = new RenderTarget(renderer, descriptor);

	return sumBuffer;

}

function startPass(renderer, target){
	let view = target.colorAttachments[0].texture.createView();

	let renderPassDescriptor = {
		colorAttachments: [{
			view, 
			loadValue: { r: 0.1, g: 0.2, b: 0.3, a: 1.0 }
		}],
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

function render(){
	Timer.setEnabled(true);

	let renderables = new Map();

	let stack = [scene.root];
	while(stack.length > 0){
		let node = stack.pop();

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
	// let dilateEnabled = Potree.settings.mode === "dilate";

	renderer.start();

	let screenbuffer = renderer.screenbuffer;
	let fbo_source = null;

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

		// render directly to screenbuffer
		let pass = startPass(renderer, screenbuffer);
		let drawstate = {renderer, camera, renderables, pass};

		if(!Potree.settings.useCompute){
			renderPoints(points, drawstate);
		}else{
			renderPointsCompute(points, drawstate);
		}

		renderPointsOctree(octrees, drawstate);
		// renderPointsOctreeBundledVBO(octrees, drawstate);

		renderer.renderDrawCommands(drawstate);

		endPass(pass);
	}else{

		// render to intermediate framebuffer
		let pass = startPass(renderer, fbo_0);
		let drawstate = {renderer, camera, renderables, pass};

		renderPointsOctree(octrees, drawstate);
		// renderPointsOctreeBundledVBO(octrees, drawstate);

		renderer.renderDrawCommands(drawstate);

		endPass(pass);

		fbo_source = fbo_0;
	}


	if(dilateEnabled){ // dilate
		let fboTarget = edlEnabled ? fbo_1 : screenbuffer;

		let pass = startPass(renderer, fboTarget);
		let drawstate = {renderer, camera, renderables, pass};

		dilate(fbo_source, drawstate);

		endPass(pass);

		fbo_source = fboTarget;
	}

	if(edlEnabled){ // EDL
		let pass = startPass(renderer, screenbuffer);
		let drawstate = {renderer, camera, renderables, pass};

		EDL(fbo_source, drawstate);

		endPass(pass);
	}


	// { // HANDLE PICKING
	// 	for(let {x, y, callback} of Potree.pickQueue){

	// 		let u = x / renderer.canvas.clientWidth;
	// 		let v = (renderer.canvas.clientHeight - y) / renderer.canvas.clientHeight;
	// 		let pos = camera.getWorldPosition();
	// 		let dir = camera.mouseToDirection(u, v);
	// 		let near = camera.near;

	// 		let window = 2;
	// 		let wh = 1;
	// 		readDepth(renderer, renderer.depthTexture, x - wh, y - wh, window, window, ({d}) => {
				
	// 			let depth = near / d;
				
	// 			dir.multiplyScalar(depth);
	// 			let position = pos.add(dir);

	// 			// console.log(position);

	// 			callback({depth, position});
	// 		});
	// 	}
	// 	Potree.pickQueue.length = 0;
	// }

	// { // MESHES
	// 	let meshes = renderables.get("Mesh") ?? [];

	// 	renderMeshes({in: meshes      , target: screenbuffer  , drawstate});
	// }

	// renderer.renderDrawCommands(drawstate);

	renderer.finish();

	Timer.frameEnd(renderer);
}


function loop(){

	stats.begin();

	update();
	render();

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

	camera = new Camera();
	controls = new OrbitControls(renderer.canvas);
	measure = new MeasureTool(renderer);

	// make things available in dev tools for debugging
	window.camera = camera;
	window.controls = controls;
	window.scene = scene;
	window.renderer = renderer;

	stats = new Stats();
	stats.showPanel(0);
	document.body.appendChild( stats.dom );

	initScene();

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

	return {scene, controls, addEventListener, removeEventListener, renderer};
}



