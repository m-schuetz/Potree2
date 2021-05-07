
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, Mesh} from "potree";
import {Renderer, Timer, EventDispatcher} from "potree";
import {drawTexture, loadImage, drawImage} from "./prototyping/textures.js";
import {geometries} from "potree";
import {Potree} from "potree";
import {loadGLB} from "potree";
import {MeasureTool} from "./interaction/measure.js";
import * as ProgressiveLoader from "./modules/progressive_loader/ProgressiveLoader.js";
import {readPixels, readDepth} from "./renderer/readPixels.js";
import {renderPoints, renderMeshes, renderPointsCompute, renderPointsOctree} from "potree";
import {dilate, EDL} from "potree";

let frame = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;

let renderer = null;
let camera = null;
let controls = null;
let measure = null;
let dbgImage = null;

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
	Potree.state.camDir = camera.getWorldDirection().toString(1);

	let size = renderer.getSize();
	camera.aspect = size.width / size.height;
	camera.updateProj();

	dispatcher.dispatch("update");

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
		octree.updateVisibility(camera);

		let numPoints = octree.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
		let numNodes = octree.visibleNodes.length;

		Potree.state.numPoints = numPoints;
		Potree.state.numNodes = numNodes;
	}

	Timer.frameStart(renderer);
	
	// let pass = renderer.start();
	renderer.start();

	let drawstate = {renderer, camera, renderables};
	let screenbuffer = renderer.screenbuffer;
	let framebuffer = renderer.getFramebuffer("point target");
	let fbo_edl = renderer.getFramebuffer("EDL target");
	let edlEnabled = Potree.settings.edlEnabled;

	framebuffer.setSize(...screenbuffer.size);
	fbo_edl.setSize(...screenbuffer.size);

	let pointTarget = edlEnabled ? fbo_edl : screenbuffer;

	if(Potree.settings.mode === "pixels"){

		renderPoints(       {in: points      , target: pointTarget , drawstate});
		renderPointsOctree( {in: octrees     , target: pointTarget , drawstate});
		
	}else if(Potree.settings.mode === "dilate"){

		renderPoints(       {in: points      , target: framebuffer  , drawstate});
		renderPointsOctree( {in: octrees     , target: framebuffer  , drawstate});

		dilate(             {in: framebuffer , target: pointTarget      , drawstate});

	}else if(Potree.settings.mode === "HQS"){

		renderPointsOctree( {in: octrees     , target: pointTarget , drawstate});
		renderPointsCompute({in: points      , target: pointTarget , drawstate});

		// dilate(             {in: framebuffer , target: screenbuffer , drawstate});
	}

	if(edlEnabled){
		EDL(                {in: fbo_edl     , target: screenbuffer , drawstate});
	}


	{ // HANDLE PICKING
		for(let {x, y, callback} of Potree.pickQueue){

			let u = x / renderer.canvas.clientWidth;
			let v = (renderer.canvas.clientHeight - y) / renderer.canvas.clientHeight;
			let pos = camera.getWorldPosition();
			let dir = camera.mouseToDirection(u, v);
			let near = camera.near;

			let window = 2;
			let wh = 1;
			readDepth(renderer, renderer.depthTexture, x - wh, y - wh, window, window, ({d}) => {
				
				let depth = near / d;
				
				dir.multiplyScalar(depth);
				let position = pos.add(dir);

				// console.log(position);

				callback({depth, position});
			});
		}
		Potree.pickQueue.length = 0;
	}

	{ // MESHES
		let meshes = renderables.get("Mesh") ?? [];

		renderMeshes({in: meshes      , target: screenbuffer  , drawstate});
	}

	// renderer.renderDrawCommands(pass, camera);
	renderer.finish();

	Timer.frameEnd(renderer);
}


function loop(){
	update();
	render();

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

	return {scene, controls, addEventListener, removeEventListener};
}
