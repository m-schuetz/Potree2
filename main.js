
import {Vector3} from "potree";
import {Scene, SceneNode, Camera, OrbitControls, Mesh} from "potree";
import {Renderer, Timer} from "potree";

import {render as renderPointsTest} from "./src/prototyping/renderPoints.js";
import {render as renderPointsCompute} from "./src/prototyping/renderPointsCompute.js";
import {drawTexture, loadImage, drawImage} from "./src/prototyping/textures.js";
import {createPointsData} from "./src/modules/geometries/points.js";
import {cube} from "./src/modules/geometries/cube.js";
import {initGUI} from "./src/prototyping/gui.js";
import {Potree} from "potree";
import {renderMesh} from "potree";
import {loadGLB} from "potree";
import {MeasureTool} from "./src/interaction/measure.js";
import {readPixels, readDepth} from "./src/renderer/readPixels.js";
import * as ProgressiveLoader from "./src/modules/progressive_loader/ProgressiveLoader.js";


import {render as renderPoints}  from "./src/potree/renderPoints.js";
import {renderDilate}  from "./src/potree/renderDilate.js";
// import {render as renderPoints}  from "./src/potree/arbitrary_attributes/renderPoints_arbitrary_attributes.js";

let frame = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;

let renderer = null;
let camera = null;
let controls = null;
let measure = null;

let scene = new Scene();
window.scene = scene;

let dbgImage = null;

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

	let pointcloud = window.pointcloud;

	if(pointcloud){
		pointcloud.showBoundingBox = guiContent["show bounding box"];
		pointcloud.pointBudget = guiContent["point budget (M)"] * 1_000_000;
		pointcloud.pointSize = guiContent["point size"];

		if(guiContent["update"]){
			let duration = pointcloud.updateVisibility(camera);

			if((frame % 60) === 0){
				guiContent["duration(update)"] = `${(duration / 1000).toFixed(1)}ms`;
			}
		}

		let numPoints = pointcloud.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
		let numNodes = pointcloud.visibleNodes.length;

		guiContent["#points"] = numPoints.toLocaleString();
		guiContent["#nodes"] = numNodes.toLocaleString();
	}


	// {
	// 	let node = scene.root.children.find(c => c.constructor.name === "Mesh");

	// 	if(node){
	// 		// node.rotation.rotate(0.9 * Math.PI / 2, new Vector3(0, 1, 0));
	// 		// node.position.set(5, 0, 3);
	// 		// node.scale.set(2, 2, 2);

	// 		// let dir = camera.mouseToDirection(u, v);
	// 		let dir = camera.getWorldDirection();

	// 		// if(dbgDepth !== Infinity){
	// 			// dir.multiplyScalar(dbgDepth);
	// 			dir.multiplyScalar(10.0);

	// 			let pos = camera.getWorldPosition().add(dir);

	// 			node.position.copy(pos);
	// 			node.rotation.makeIdentity();
	// 			node.updateWorld();

	// 		// }


	// 	}
	// }
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

	let target = null;

	Timer.frameStart(renderer);
	
	if(guiContent["mode"] === "HQS"){
		let points = renderables.get("Points") ?? [];

		for(let point of points){
			target = renderPointsCompute(renderer, point, camera);
		}
	}

	let pass = renderer.start();

	if(guiContent["mode"] === "dilate"){
		let octrees = renderables.get("PointCloudOctree") ?? [];

		renderDilate(renderer, pass, octrees, camera);
	}

	if(dbgImage){
		drawImage(renderer, pass, dbgImage, 0.1, 0.1, 0.1, 0.1);
	}

	if(target){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}

	if(guiContent["mode"] === "pixels"){
		let points = renderables.get("Points") ?? [];

		for(let point of points){
			renderPointsTest(renderer, pass, point, camera);
		}

		let octrees = renderables.get("PointCloudOctree") ?? [];
		for(let octree of octrees){
			renderPoints(renderer, pass, octree, camera);
		}
	}

	// if(pointcloud.pointSize === 1){
	// 	renderPoints(renderer, pass, pointcloud, camera);
	// }else{
	// 	renderQuads(renderer, pass, pointcloud, camera);
	// }

	{
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

				callback({depth, position});
			});
		}
		Potree.pickQueue.length = 0;
	}

	{ // MESHES
		let meshes = renderables.get("Mesh") ?? [];

		for(let mesh of meshes){
			renderMesh(renderer, pass, mesh, camera, renderables);
		}
	}

	renderer.renderDrawCommands(pass, camera);
	renderer.finish(pass);

	Timer.frameEnd(renderer);

}

function loop(){
	update();
	render();

	requestAnimationFrame(loop);
}

async function main(){

	initGUI();

	renderer = new Renderer();
	window.renderer = renderer;

	await renderer.init();

	camera = new Camera();
	controls = new OrbitControls(renderer.canvas);
	measure = new MeasureTool(renderer);

	window.camera = camera;
	window.controls = controls;

	camera.fov = 60;

	controls.set({
		yaw: -0.2,
		pitch: 0.8,
		radius: 10,
	});

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

	Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {
		// controls.set({
		// 	radius: 7,
		// 	yaw: -0.86,
		// 	pitch: 0.51,
		// 	pivot: [-0.22, -0.01, 3.72],
		// });

		scene.root.children.push(pointcloud);
	});

	// Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
	// 	scene.root.children.push(pointcloud);
	// });

	controls.set({
		radius: 13.8,
		yaw: -0.66,
		pitch: 0.37,
		pivot: [-0.022888880829764084, -0.12292264906406908, 5.322860838969788],
	});

	// Potree.load("./resources/pointclouds/ca13/metadata.json").then(pointcloud => {

	// 	// controls.zoomTo(pointcloud);
	// 	controls.set({
	// 		yaw: -1.1,
	// 		pitch: 0.37,
	// 		radius: 406,
	// 		pivot: [696743.76, 3919073.53, 37.68],
	// 	});

	// 	scene.root.children.push(pointcloud);
	// });

	loadImage("./resources/images/background.jpg").then(image => {
		dbgImage = image;
	});

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

	{
		let mesh = new Mesh("cube", cube);
		mesh.scale.set(0.5, 0.5, 0.5);

		scene.root.children.push(mesh);
	}


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

}

main();
