
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, Mesh} from "potree";
import {Renderer, Timer} from "potree";
import {drawTexture, loadImage, drawImage} from "./prototyping/textures.js";
import {geometries} from "potree";
import {initGUI} from "./prototyping/gui.js";
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

let scene = new Scene();

function initScene(){

	// camera.fov = 60;

	// default, if not overriden later on
	// controls.set({
	// 	yaw: -0.2,
	// 	pitch: 0.8,
	// 	radius: 10,
	// });


	// {
	// 	let points = createPointsData(6_000);
	// 	scene.root.children.push(points);

	// 	controls.set({
	// 		radius: 7.8,
	// 		yaw: 5.66,
	// 		pitch: 0.7,
	// 		pivot: [0.2094438030078714, -0.23336995166449773, 0.2092711916242336],
	// 	});

	// }

	// Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {
	// 	// controls.set({
	// 	// 	radius: 7,
	// 	// 	yaw: -0.86,
	// 	// 	pitch: 0.51,
	// 	// 	pivot: [-0.22, -0.01, 3.72],
	// 	// });

	// 	scene.root.children.push(pointcloud);
	// });

	// let url = "./resources/pointclouds/heidentor/metadata.json";
	// Potree.load(url, {name: "Heidentor"}).then(pointcloud => {
	// 	scene.root.children.push(pointcloud);

	// 	controls.set({
	// 		radius: 26.8, yaw: -4.2, pitch: 0.31,
	// 		pivot: [-0.182792265881022, 1.9724050351418307, 5.693598313985278],
	// 	});
	// });

	// Potree.load("./resources/pointclouds/ca13/metadata.json", {name: "CA13"}).then(pointcloud => {
	// 	scene.root.children.push(pointcloud);

	// 	// controls.zoomTo(pointcloud);
	// 	controls.set({
	// 		yaw: -1.1, pitch: 0.37, radius: 406,
	// 		pivot: [696743.76, 3919073.53, 37.68],
	// 	});

	// 	// guiContent["point budget (M)"] = 0.0015;
	// 	// controls.set({
	// 	// 	yaw: 0.13474178403755852,
	// 	// 	pitch: 0.8641724941724933,
	// 	// 	radius: 116542.60616174094,
	// 	// 	pivot:  [696743.76, 3919073.53, 37.68],
	// 	// });

	// });

	// loadGLB("./resources/models/anita_mui.glb").then(node => {
	// // loadGLB("./resources/models/lion.glb").then(node => {
	// 	scene.root.children.push(node);

	// 	node.rotation.rotate(0.9 * Math.PI / 2, new Vector3(0, 1, 0));
	// 	node.position.set(5, 0, 3);
	// 	node.scale.set(2, 2, 2);
	// 	node.updateWorld();

	// 	// controls.set({
	// 	// 	yaw: -9.2375,
	// 	// 	pitch: 0.2911333847340012,
	// 	// 	radius: 3.649930853878021,
	// 	// 	pivot:  [0.3169157776176301, -0.055293688684424885, 2.2],
	// 	// });

	// 	// controls.zoomTo(node);
	// });

	// loadImage("./resources/images/background.jpg").then(image => {
	// 	dbgImage = image;
	// });

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
		guiContent["fps"] = Math.floor(fps).toLocaleString();
	}

	frame++;
	framesSinceLastCount++;

	controls.update();
	camera.world.copy(controls.world);

	camera.updateView();
	guiContent["cam.pos"] = camera.getWorldPosition().toString(1);
	guiContent["cam.dir"] = camera.getWorldDirection().toString(1);

	let size = renderer.getSize();
	camera.aspect = size.width / size.height;
	camera.updateProj();

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

		octree.showBoundingBox = guiContent["show bounding box"];
		octree.pointBudget = guiContent["point budget (M)"] * 1_000_000;
		octree.pointSize = guiContent["point size"];
		octree.updateVisibility(camera);

		let numPoints = octree.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
		let numNodes = octree.visibleNodes.length;

		guiContent["#points"] = numPoints.toLocaleString();
		guiContent["#nodes"] = numNodes.toLocaleString();
	}

	Timer.frameStart(renderer);
	
	// let pass = renderer.start();
	renderer.start();

	let drawstate = {renderer, camera, renderables};
	let screenbuffer = renderer.screenbuffer;
	let framebuffer = renderer.getFramebuffer("point target");
	let fbo_edl = renderer.getFramebuffer("EDL target");

	framebuffer.setSize(...screenbuffer.size);
	fbo_edl.setSize(...screenbuffer.size);

	// let framebuffer1 = renderer.getFramebuffer("point target 1");
	// framebuffer1.setSize(...screenbuffer.size);

	// if(dbgImage){
	// 	drawImage(renderer, pass, dbgImage, 0.1, 0.1, 0.1, 0.1);
	// }

	if(guiContent["mode"] === "pixels"){

		renderPoints(       {in: points      , target: screenbuffer , drawstate});
		renderPointsOctree( {in: octrees     , target: screenbuffer , drawstate});
		
	}else if(guiContent["mode"] === "dilate"){

		renderPoints(       {in: points      , target: framebuffer  , drawstate});
		// Timer.timestampSep(renderer,"dbg-start");
		renderPointsOctree( {in: octrees     , target: framebuffer  , drawstate});
		// Timer.timestampSep(renderer,"dbg-end");

		// dilate(             {in: framebuffer , target: framebuffer1 , drawstate});
		// dilate(             {in: framebuffer1, target: screenbuffer , drawstate});

		// dilate(             {in: framebuffer , target: screenbuffer , drawstate});

		dilate(             {in: framebuffer , target: fbo_edl      , drawstate});
		EDL(                {in: fbo_edl     , target: screenbuffer , drawstate});

	}else if(guiContent["mode"] === "HQS"){

		renderPointsOctree( {in: octrees     , target: screenbuffer , drawstate});
		renderPointsCompute({in: points      , target: screenbuffer , drawstate});

		// dilate(             {in: framebuffer , target: screenbuffer , drawstate});
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

	initGUI();

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

	return {scene, controls};
}
