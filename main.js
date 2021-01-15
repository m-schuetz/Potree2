

import {Renderer} from "./src/renderer/Renderer.js";
import {Camera} from "./src/scene/Camera.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {Vector3, Matrix4} from "./src/math/math.js";

import {Potree} from "./src/Potree.js";

import {render as renderQuads}  from "./src/potree/renderQuads.js";
import {render as renderPoints}  from "./src/potree/renderPoints.js";
import {renderDilate}  from "./src/potree/renderDilate.js";
import {renderAtomic}  from "./src/potree/renderAtomic.js";
import {renderAtomicDilate}  from "./src/potree/renderAtomicDilate.js";
import {drawTexture} from "./src/prototyping/textures.js";

import * as ProgressiveLoader from "./src/modules/progressive_loader/ProgressiveLoader.js";

import * as dat from "./libs/dat.gui/dat.gui.module.js";

let frame = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;

let renderer = null;
let camera = null;
let controls = null;
let progress = null;

let boxes = [];

let gui = null;
let guiContent = {
	"#points": "0",
	"#nodes": "0",
	"fps": "0",
	"duration(update)": "0",
	"camera": "",

	"show bounding box": true,
	"mode": "points/quads",
	//"mode": "points/atomic",
	// "mode": "points/atomic/dilate",
	"point budget (M)": 3,
	"point size": 1,
	"update": true,
};
window.guiContent = guiContent;


function initGUI(){

	gui = new dat.GUI();
	
	{
		let stats = gui.addFolder("stats");
		stats.open();
		stats.add(guiContent, "#points").listen();
		stats.add(guiContent, "#nodes").listen();
		stats.add(guiContent, "fps").listen();
		stats.add(guiContent, "duration(update)").listen();
		stats.add(guiContent, "camera").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		input.add(guiContent, "mode", [
			"points/quads", 
			"points/dilate", 
			// "points/atomic", 
			// "points/atomic/dilate"
			]);
		input.add(guiContent, "show bounding box");
		input.add(guiContent, "update");

		// slider
		input.add(guiContent, 'point budget (M)', 0.1, 5);
		input.add(guiContent, 'point size', 1, 5);
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
	guiContent["camera"] = camera.getWorldPosition().toString(1);

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
}

function render(){

	let pointcloud = window.pointcloud;
	let target = null;
	let texture = null;

	let shouldDrawTarget = false;
	if(pointcloud && guiContent["mode"] === "points/dilate"){
		target = renderDilate(renderer, pointcloud, camera);
		target = target.colorAttachments[0].texture;

		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "points/atomic"){
		target = renderAtomic(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "points/atomic/dilate"){
		target = renderAtomicDilate(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}


	let pass = renderer.start();

	// draw point cloud
	if(pointcloud && guiContent["mode"] === "points/quads"){

		if(pointcloud.pointSize === 1){
			renderPoints(renderer, pass, pointcloud, camera);
		}else{
			renderQuads(renderer, pass, pointcloud, camera);
		}
	}else if(pointcloud && guiContent["mode"] === "points/atomic"){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}else if(pointcloud && guiContent["mode"] === "points/atomic/dilate"){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}else if(shouldDrawTarget){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}

	{ // draw xyz axes
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(255, 0, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, 2, 0), new Vector3(0, 255, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, 0, 2), new Vector3(0, 0, 255));
	}

	if(guiContent["show bounding box"]){ // draw boxes
		for(let box of boxes){
			let position = box.center();
			let size = box.size();
			let color = new Vector3(255, 255, 0);

			renderer.drawBoundingBox(position, size, color);
		}
	}

	{
		if(progress?.octree?.visibleNodes.length > 0){
			// console.log(progress);
			// renderPoints(renderer, pass, progress.octree, camera);

			let numPoints = progress.octree.visibleNodes.reduce( (a, i) => a + i.geometry.numElements, 0);
			let text = `${(numPoints / 1_000_000).toFixed(1)}M points`;
			// document.getElementById("big_message").innerText = text;
		}
		// renderPoints(renderer, pass, pointcloud, camera);
	}
	
	renderer.renderDrawCommands(pass, camera);
	renderer.finish(pass);

}

function loop(){
	update();
	render();

	requestAnimationFrame(loop);
}

async function run(){

	initGUI();

	renderer = new Renderer();
	window.renderer = renderer;

	await renderer.init();

	camera = new Camera();
	controls = new OrbitControls(renderer.canvas);

	window.camera = camera;
	window.controls = controls;

	camera.fov = 60;

	{
		let element = document.getElementById("canvas");
		ProgressiveLoader.install(element, (e) => {
			//console.log(e.boxes);
			boxes = e.boxes;

			progress = e.progress;
			window.progress = progress;

			console.log(progress);

			let pivot = progress.boundingBox.center();
			pivot.z = 0.8 * progress.boundingBox.min.z + 0.2 * progress.boundingBox.max.z;
			controls.pivot.copy(pivot);
			controls.radius = progress.boundingBox.size().length() * 0.7;

			window.pointcloud = progress.octree;
		});
	}

	// Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {

	// 	controls.radius = 10;
	// 	controls.yaw = -Math.PI / 6;
	// 	controls.pitch = Math.PI / 5;

	// 	pointcloud.updateVisibility(camera);
	// 	pointcloud.position.set(-0.9, 0.1, -5);
	// 	pointcloud.updateWorld();
	// 	window.pointcloud = pointcloud;

	// });

	// Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
	// 	controls.radius = 20;
	// 	controls.yaw = 2.7 * Math.PI / 4;
	// 	controls.pitch = Math.PI / 6;
	
	// 	pointcloud.updateVisibility(camera);
	// 	pointcloud.position.set(3, -3, -6)
	// 	pointcloud.updateWorld();
	// 	window.pointcloud = pointcloud;
	// });

	controls.radius = 20;
	controls.yaw = -0.2;
	controls.pitch = 0.8;
	camera.updateProj();

	// Potree.load("./resources/pointclouds/eclepens/metadata.json").then(pointcloud => {
	// 	controls.radius = 700;
	// 	controls.yaw = -0.2;
	// 	controls.pitch = 0.8;
	// 	camera.near = 1;
	// 	camera.far = 10_000;
	// 	camera.updateProj();
	
	// 	pointcloud.updateVisibility(camera);
	// 	// pointcloud.position.set(400, -300, -6)
	// 	// pointcloud.position.copy(pointcloud.boundingBox.min);
	// 	// pointcloud.updateWorld();
	// 	window.pointcloud = pointcloud;
	// });


	Potree.load("./resources/pointclouds/CA13/metadata.json").then(pointcloud => {
		camera.near = 0.5;
		camera.far = 100_000;

		controls.radius = 2_400;
		controls.yaw = 0.03437500000000017;
		controls.pitch = 0.6291441788743247;
		controls.pivot.set(694698.4629456067, 3916428.1845130883, -15.72393889322449);

		camera.updateProj();
	
		pointcloud.updateVisibility(camera);

		window.pointcloud = pointcloud;
	});

	requestAnimationFrame(loop);

}


run();
