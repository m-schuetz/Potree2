

import {Renderer} from "./src/renderer/Renderer.js";
import {Camera} from "./src/scene/Camera.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {Vector3, Matrix4} from "./src/math/math.js";

import {Potree} from "./src/Potree.js";

import {render as renderQuads}  from "./src/potree/renderQuads.js";
import {render as renderPoints}  from "./src/potree/renderPoints.js";
import {renderFill}  from "./src/potree/renderFill.js";
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

	"show bounding box": false,
	"mode": "points/dilute",
	"point budget (M)": 3,
	"point size": 1,
	"update": true,
};


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

		// input.add(guiContent, "primitive", ["points", "quads"]);
		input.add(guiContent, "mode", ["points/quads", "points/dilute"]);
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
	// mat4.copy(camera.world, controls.world);
	// mat4.set(camera.world, ...controls.world.elements);
	camera.world.copy(controls.world);

	{
		// let flip = mat4.create();
		// mat4.set(flip,
		// 	1, 0, 0, 0,
		// 	0, 0, 1, 0,
		// 	0, -1, 0, 0,
		// 	0, 0, 0, 1,
		// );

		let flip = new Matrix4().set(
			1, 0, 0, 0,
			0, 0, -1, 0,
			0, 1, 0, 0,
			0, 0, 0, 1,
		);

		camera.world.multiplyMatrices(flip, camera.world);

		// mat4.multiply(camera.world, flip, camera.world);
	}
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

	if(pointcloud && guiContent["mode"] === "points/dilute"){
		target = renderFill(renderer, pointcloud, camera);
	}


	let pass = renderer.start();

	// draw point cloud
	if(pointcloud && guiContent["mode"] === "points/quads"){

		if(pointcloud.pointSize === 1){
			renderPoints(renderer, pass, pointcloud, camera);
		}else{
			renderQuads(renderer, pass, pointcloud, camera);
		}
	}else if(pointcloud && guiContent["mode"] === "points/dilute"){
		drawTexture(renderer, pass, target.colorAttachments[0].texture, 
			0, 0, 1, 1);
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
			renderPoints(renderer, pass, progress.octree, camera);
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

	Potree.load("./resources/pointclouds/eclepens/metadata.json").then(pointcloud => {
		controls.radius = 700;
		controls.yaw = -0.2;
		controls.pitch = 0.8;
		camera.updateProj();
	
		pointcloud.updateVisibility(camera);
		// pointcloud.position.set(400, -300, -6)
		pointcloud.updateWorld();
		window.pointcloud = pointcloud;
	});


	// Potree.load("./resources/pointclouds/CA13/metadata.json").then(pointcloud => {
	// 	camera.near = 0.5;
	// 	camera.far = 20_000;
	// 	// controls.radius = 1000;
	// 	// controls.yaw = -0.2;
	// 	// controls.pitch = Math.PI / 5;

	// 	controls.radius = 1000;
	// 	controls.yaw = -0.2;
	// 	controls.pitch = 0;
	// 	// controls.pivot.set(643431, 3889087, -2.7);
	// 	camera.updateProj();
	
	// 	pointcloud.updateVisibility(camera);
	// 	// pointcloud.position.set(-643431, -3889087, 0);
	// 	pointcloud.updateWorld();
	// 	window.pointcloud = pointcloud;
	// });

	requestAnimationFrame(loop);

}


run();
