

import {Renderer} from "./src/renderer/Renderer.js";
import {Camera} from "./src/scene/Camera.js";
import {Scene} from "./src/scene/Scene.js";
import {PointLight} from "./src/scene/PointLight.js";
import {Mesh} from "./src/modules/mesh/Mesh.js";
import {PhongMaterial} from "./src/modules/mesh/PhongMaterial.js";
import {NormalMaterial} from "./src/modules/mesh/NormalMaterial.js";
import {render as renderMesh} from "./src/modules/mesh/renderMesh.js";
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {Vector3, Matrix4} from "./src/math/math.js";
import {Geometry} from "./src/core/Geometry.js";
import {cube, createWave} from "./src/prototyping/cube.js";

import {Potree} from "./src/Potree.js";

import {render as renderQuads}  from "./src/potree/renderQuads.js";
import {render as renderPoints}  from "./src/potree/renderPoints.js";
import {renderDilate}  from "./src/potree/renderDilate.js";
import {renderAtomic}  from "./src/potree/renderAtomic.js";
import {renderAtomicDilate} from "./src/potree/render_compute_dilate/render_compute_dilate.js";
import {renderComputeLoop} from "./src/potree/render_compute_loop/render_compute_loop.js";
import {renderComputeNoDepth} from "./src/potree/render_compute_no_depth/render_compute_no_depth.js";
import {render as renderComputePacked} from "./src/potree/render_compute_packed/render_compute_packed.js";
import {render as renderComputeXRay} from "./src/potree/render_compute_xray/render_compute_xray.js";
import {render as renderProgressive} from "./src/potree/render_progressive/render_progressive.js";
import {drawTexture} from "./src/prototyping/textures.js";
import * as Timer from "./src/renderer/Timer.js";

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

let scene = new Scene();

let boxes = [];

let gui = null;
let guiContent = {
	"#points": "0",
	"#nodes": "0",
	"fps": "0",
	"duration(update)": "0",
	// "timings": "",
	"camera": "",

	"show bounding box": false,
	"mode": "points/quads",
	//"mode": "points/atomic",
	// "mode": "compute/dilate",
	// "mode": "compute/xray",
	// "mode": "compute/packed",
	// "mode": "compute/loop",
	// "mode": "compute/no_depth",
	// "mode": "progressive",
	"point budget (M)": 2,
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
		// stats.add(guiContent, "timings").listen();
		stats.add(guiContent, "camera").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		input.add(guiContent, "mode", [
			"points", 
			"points/quads", 
			"points/dilate", 
			"points/atomic",
			"compute/dilate",
			"compute/loop",
			"compute/no_depth",
			"compute/packed",
			"compute/xray",
			"progressive",
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

	// renderer.render(scene, camera);

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
			stack.push(child);
		}
	}

	// console.log(renderables);

	let pointcloud = window.pointcloud;
	let target = null;

	Timer.frameStart(renderer);

	let shouldDrawTarget = false;
	if(pointcloud && guiContent["mode"] === "points/dilate"){
		target = renderDilate(renderer, pointcloud, camera);
		target = target.colorAttachments[0].texture;

		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "points/atomic"){
		target = renderAtomic(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "compute/dilate"){
		target = renderAtomicDilate(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "compute/loop"){
		target = renderComputeLoop(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "compute/packed"){
		target = renderComputePacked(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "compute/no_depth"){
		target = renderComputeNoDepth(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "compute/xray"){
		target = renderComputeXRay(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}else if(pointcloud && guiContent["mode"] === "progressive"){
		target = renderProgressive(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}


	let pass = renderer.start();

	// draw point cloud
	if(pointcloud && guiContent["mode"] === "points"){
		renderPoints(renderer, pass, pointcloud, camera);
	}else if(pointcloud && guiContent["mode"] === "points/quads"){

		if(pointcloud.pointSize === 1){
			renderPoints(renderer, pass, pointcloud, camera);
		}else{
			renderQuads(renderer, pass, pointcloud, camera);
		}
	}else if(shouldDrawTarget){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}

	

	{ // draw xyz axes
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(255, 0, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, 2, 0), new Vector3(0, 255, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, 0, 2), new Vector3(0, 0, 255));
	}

	// draw boxes
	if(guiContent["show bounding box"]){ 
		for(let box of boxes){
			let position = box.center();
			let size = box.size();
			let color = new Vector3(255, 255, 0);

			renderer.drawBoundingBox(position, size, color);
		}
	}

	{
		let meshes = renderables.get("Mesh");

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

	controls.radius = 20;
	controls.yaw = -0.2;
	controls.pitch = 0.8;
	camera.updateProj();

	Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {
		controls.pivot.set(0.46849801014552056, -0.5089652605462774, 4.694897729016537);
		controls.pitch = 0.3601621061369527;
		controls.yaw = -0.610317525598302;
		controls.radius = 6.3;

		window.pointcloud = pointcloud;
	});

	// Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
	// 	controls.radius = 20;
	// 	controls.yaw = 2.7 * Math.PI / 4;
	// 	controls.pitch = Math.PI / 6;
	
	// 	pointcloud.updateVisibility(camera);
	// 	window.pointcloud = pointcloud;
	// });

	

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


	// Potree.load("./resources/pointclouds/CA13/metadata.json").then(pointcloud => {
	// // Potree.load("http://5.9.65.151/mschuetz/potree/resources/pointclouds/opentopography/CA13_2.0.2_brotli/metadata.json").then(pointcloud => {
	// 	camera.near = 0.5;
	// 	camera.far = 100_000;

	// 	controls.radius = 2_400;
	// 	controls.yaw = 0.03437500000000017;
	// 	controls.pitch = 0.6291441788743247;
	// 	controls.pivot.set(694698.4629456067, 3916428.1845130883, -15.72393889322449);

	// 	camera.updateProj();
	
	// 	pointcloud.updateVisibility(camera);

	// 	scene.root.children.push(pointcloud);

	// 	window.pointcloud = pointcloud;
	// });

	{
		let geometry = new Geometry();
		let ref = createWave();

		geometry.buffers = ref.buffers;
		geometry.numElements = ref.vertexCount;
		geometry.indices = ref.indices;

		let mesh = new Mesh("mesh", geometry);
		scene.root.children.push(mesh);

		mesh.material = new PhongMaterial();
	}

	{
		let light = new PointLight("pointlight");
		light.position.set(0, 0, 5);

		scene.root.children.push(light);
	}

	requestAnimationFrame(loop);

}


run();
