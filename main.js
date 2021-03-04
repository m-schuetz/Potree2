

import * as dat from "./libs/dat.gui/dat.gui.module.js";
import { Box3, Vector3, Frustum, Matrix4, Ray } from "potree";
import { Potree } from "potree";
import { load as loadGLB } from "./src/misc/GLBLoader.js";
import { render as renderMesh } from "./src/modules/mesh/renderMesh.js";
import * as ProgressiveLoader from "./src/modules/progressive_loader/ProgressiveLoader.js";
import { OrbitControls } from "./src/navigation/OrbitControls.js";
import { render as renderPointsArbitraryAttributes } from "./src/potree/arbitrary_attributes/renderPoints_arbitrary_attributes.js";
import { renderAtomic } from "./src/potree/renderAtomic.js";
import { renderDilate } from "./src/potree/renderDilate.js";
import { render as renderPoints } from "./src/potree/renderPoints.js";
import { render as renderQuads } from "./src/potree/renderQuads.js";
import { renderAtomicDilate } from "./src/potree/render_compute_dilate/render_compute_dilate.js";
import { renderComputeLoop } from "./src/potree/render_compute_loop/render_compute_loop.js";
import { renderComputeNoDepth } from "./src/potree/render_compute_no_depth/render_compute_no_depth.js";
import { render as renderComputePacked } from "./src/potree/render_compute_packed/render_compute_packed.js";
import { render as renderComputeXRay } from "./src/potree/render_compute_xray/render_compute_xray.js";
import { render as renderProgressive } from "./src/potree/render_progressive/render_progressive.js";
import { render as renderProgressiveSimple } from "./src/potree/render_progressive_simple/render_progressive.js";
import { drawTexture } from "./src/prototyping/textures.js";
import { Renderer } from "./src/renderer/Renderer.js";
import * as Timer from "./src/renderer/Timer.js";
import { Camera } from "./src/scene/Camera.js";
import { PointLight } from "./src/scene/PointLight.js";
import { Scene } from "./src/scene/Scene.js";
import { SceneNode } from "./src/scene/SceneNode.js";
import {cube} from "./src/modules/geometries/cube.js";
import { NormalMaterial } from "./src/modules/mesh/NormalMaterial.js";
import { PhongMaterial } from "./src/modules/mesh/PhongMaterial.js";
import {Geometry} from "./src/core/Geometry.js";
import {Mesh} from "./src/modules/mesh/Mesh.js";

import {Points, render as renderPointsTest} from "./src/prototyping/renderPoints.js";

window.math = {Vector3, Frustum, Matrix4, Ray, Box3};

let frame = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;

let renderer = null;
let camera = null;
let controls = null;

let scene = new Scene();
window.scene = scene;

let gui = null;
let guiContent = {

	// INFOS
	"#points": "0",
	"#nodes": "0",
	"fps": "0",
	"duration(update)": "0",
	// "timings": "",
	"camera": "",


	// INPUT
	"show bounding box": false,
	"mode": "points",
	// "mode": "points/quads",
	//"mode": "points/atomic",
	// "mode": "compute/dilate",
	// "mode": "compute/xray",
	// "mode": "compute/packed",
	// "mode": "compute/loop",
	// "mode": "compute/no_depth",
	// "mode": "progressive/simple",
	"attribute": "rgba",
	"point budget (M)": 10,
	"point size": 3,
	"update": true,

	// COLOR ADJUSTMENT
	"scalar min": 0,
	"scalar max": 2 ** 16,
	"gamma": 1,
	"brightness": 0,
	"contrast": 0,
};
window.guiContent = guiContent;
let guiAttributes = null;
let guiScalarMin = null;
let guiScalarMax = null;


function initGUI(){

	gui = new dat.GUI();
	window.gui = gui;
	
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
			"progressive/simple",
			]);
		input.add(guiContent, "show bounding box");
		input.add(guiContent, "update");
		guiAttributes = input.add(guiContent, "attribute", ["rgba"]).listen();
		window.guiAttributes = guiAttributes;

		// slider
		input.add(guiContent, 'point budget (M)', 0.01, 5);
		input.add(guiContent, 'point size', 1, 5);
	}

	{
		let input = gui.addFolder("Color Adjustments");
		input.open();

		guiScalarMin = input.add(guiContent, 'scalar min', 0, 2 ** 16).listen();
		guiScalarMax = input.add(guiContent, 'scalar max', 0, 2 ** 16).listen();
		input.add(guiContent, 'gamma', 0, 2).listen();
		input.add(guiContent, 'brightness', -1, 1).listen();
		input.add(guiContent, 'contrast', -1, 1).listen();
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

	let renderables = new Map();

	camera.near = Math.max(controls.radius / 100, 0.001);
	camera.far = Math.max(controls.radius * 2, 100_000);

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
	}else if(pointcloud && guiContent["mode"] === "progressive/simple"){
		target = renderProgressiveSimple(renderer, pointcloud, camera);
		shouldDrawTarget = true;
	}

	// Timer.timestampSep(renderer, "000");


	let pass = renderer.start();

	// Timer.timestamp(pass.passEncoder, "010");

	// draw point cloud
	if(pointcloud && guiContent["mode"] === "points"){
		renderPointsArbitraryAttributes(renderer, pass, pointcloud, camera);
		// renderPoints(renderer, pass, pointcloud, camera);
	}else if(pointcloud && guiContent["mode"] === "points/quads"){

		if(pointcloud.pointSize === 1){
			renderPoints(renderer, pass, pointcloud, camera);
		}else{
			renderQuads(renderer, pass, pointcloud, camera);
		}
	}else if(shouldDrawTarget){
		drawTexture(renderer, pass, target, 0, 0, 1, 1);
	}

	// Timer.timestamp(pass.passEncoder, "020");
	

	{ // draw xyz axes
		let length = controls.radius / 10.0;
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(length, 0, 0), new Vector3(255, 0, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, length, 0), new Vector3(0, 255, 0));
		renderer.drawLine(new Vector3(0, 0, 0), new Vector3(0, 0, length), new Vector3(0, 0, 255));
	}

	// { // draw ground grid

	// 	let size = 10;
	// 	//let step = Math.floor(controls.radius / 10);
	// 	let step = 10 ** Math.floor(Math.log10(controls.radius / 5))
	// 	step = Math.max(1, step);


	// 	for(let i = -size / 2; i <= size / 2; i++){
	// 		for(let j = -size / 2; j <= size / 2; j++){

	// 			renderer.drawLine(
	// 				new Vector3(step * i, -step * j, 0), 
	// 				new Vector3(step * i,  step * j, 0), 
	// 				new Vector3(150, 150, 150));

	// 			renderer.drawLine(
	// 				new Vector3(-step * j, step * i, 0), 
	// 				new Vector3( step * j, step * i, 0), 
	// 				new Vector3(150, 150, 150));

	// 		}
	// 	}
	// }

	{ // MESHES
		let meshes = renderables.get("Mesh") ?? [];

		for(let mesh of meshes){
			renderMesh(renderer, pass, mesh, camera, renderables);
		}
	}

	{
		let points = renderables.get("Points") ?? [];

		for(let point of points){
			renderPointsTest(renderer, pass, point, camera);
		}
	}

	{ // PROGRESSIVE POINT CLOUDS
		let progressives = renderables.get("ProgressivePointCloud") ?? [];

		for(let progressive of progressives){

			progressive.update(renderer, camera);

			let boxes = progressive?.renderables?.boxes ?? [];
			let boundingBoxes = progressive?.renderables?.boundingBoxes ?? [];

			for(let box of boxes){
				let position = box.boundingBox.center();
				let size = box.boundingBox.size();
				let color = box.color;
				
				renderer.drawBox(position, size, color);
			}

			for(let box of boundingBoxes){
				let position = box.boundingBox.center();
				let size = box.boundingBox.size();
				let color = box.color;
				
				renderer.drawBoundingBox(position, size, color);
			}
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
		ProgressiveLoader.install(element, {
			progressivePointcloudAdded: (e) => {
				
				scene.root.children.push(e.node);

				// controls.set({
				// 	pitch: 0.6010794140323822,
				// 	pivot:  {x: 694417.4634171372, y: 3916280.3929701354, z: 153.10151835027543},
				// 	radius: 24219.941123255907,
				// 	yaw: -12.38281250000001,
				// });
				// controls.set({
				// 	pitch: 0.696849565668457,
				// 	pivot: {x: 694417.4634171372, y: 3916280.3929701354, z: 153.10151835027543},
				// 	radius: 67475.84991112133,
				// 	yaw: -12.35468750000004,
				// });
			}
		});
	}

	controls.set({
		yaw: -0.2,
		pitch: 0.8,
		radius: 10,
	});


	function setPointcloud(pointcloud){
		let attributes = pointcloud.loader.attributes.attributes.map(b => b.name).filter(n => n !== "position");

		let onChange = () => {
			let attributeName = guiContent.attribute;
			let attribute = pointcloud.loader.attributes.attributes.find(a => a.name === attributeName);
			let range = attribute.range;

			let getRangeVal = (val) => {
				if(typeof val === "number"){
					return val;
				}else{
					return Math.max(...val);
				}
			};

			let low = getRangeVal(range[0]);
			let high = getRangeVal(range[1]);

			if(attributeName === "rgba"){
				low = 0;
				high = high > 255 ? (2 ** 16 - 1) : 255;
			}

			if(attributeName === "intensity"){
				guiContent["gamma"] = 0.5;
				guiContent["brightness"] = 0;
				guiContent["contrast"] = 0;
			}else{

				guiContent["gamma"] = 1;
				guiContent["brightness"] = 0;
				guiContent["contrast"] = 0;
			}

			guiContent["scalar min"] = low;
			guiContent["scalar max"] = high;

			guiScalarMin = guiScalarMin.min(low);
			guiScalarMin = guiScalarMin.max(high);

			guiScalarMax = guiScalarMax.min(low);
			guiScalarMax = guiScalarMax.max(high);
		};

		guiAttributes = guiAttributes.options(attributes).setValue("rgba").onChange(onChange);
		onChange();
	}

	
	// Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {

	// 	// controls.set({
	// 	// 	pivot: [0.46849801014552056, -0.5089652605462774, 4.694897729016537],
	// 	// 	pitch: 0.3601621061369527,
	// 	// 	yaw: -0.610317525598302,
	// 	// 	radius: 6.3,
	// 	// });

	// 	controls.set({
	// 		pitch: 0.44635524260941817,
	// 		pivot: {x: 0.5546404301815215, y: -1.1738194865078735, z: 0.902966295867063},
	// 		radius: 2.2613368972380035,
	// 		yaw: -0.7954737755983013,
	// 	});

	// 	pointcloud.scale.set(0.4, 0.4, 0.4)
	// 	pointcloud.position.set(0, -2, 0);
	// 	pointcloud.updateWorld()

	// 	window.pointcloud = pointcloud;
	// });

	{
		// let n = 10_000_000;
		// let position = new Float32Array(3 * n);
		// let color = new Uint8Array(4 * n);
		// for(let i = 0; i < n; i++){
		// 	let x = 2.0 * Math.random() - 1.0;
		// 	let y = 2.0 * Math.random() - 1.0;
		// 	let z = 2.0 * Math.random() - 1.0;

		// 	position[3 * i + 0] = x;
		// 	position[3 * i + 1] = y;
		// 	position[3 * i + 2] = z;

		// 	color[4 * i + 0] = 255 * Math.random();
		// 	color[4 * i + 1] = 255 * Math.random();
		// 	color[4 * i + 2] = 255 * Math.random();
		// 	color[4 * i + 3] = 255;
		// }

		let cells = 10_000;
		let n = cells * cells;
		let position = new Float32Array(3 * n);
		let color = new Uint8Array(4 * n);
		let k = 0;
		for(let i = 0; i < cells; i++){
		for(let j = 0; j < cells; j++){

			let u = 2.0 * (i / cells) - 1.0;
			let v = 2.0 * (j / cells) - 1.0;

			let x = 2 * u;
			let y = 2 * v;
			let z = 0;

			position[3 * k + 0] = x;
			position[3 * k + 1] = y;
			position[3 * k + 2] = z;

			color[4 * k + 0] = 255 * u;
			color[4 * k + 1] = 255 * v;
			color[4 * k + 2] = 0;
			color[4 * k + 3] = 255;

			k++;
		}
		}


		let numElements = n;
		let buffers = [
			{
				name: "position",
				buffer: position,
			},{
				name: "rgba",
				buffer: color,
			}
		];
		let geometry = new Geometry({numElements, buffers});
		let points = new Points();
		points.geometry = geometry;
		
		scene.root.children.push(points);

	}

	// Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
	// 	controls.radius = 20;
	// 	controls.yaw = 2.7 * Math.PI / 4;
	// 	controls.pitch = Math.PI / 6;
	
	// 	pointcloud.updateVisibility(camera);
	// 	window.pointcloud = pointcloud;
	// });

	// Potree.load("./resources/pointclouds/eclepens/metadata.json").then(pointcloud => {

	// 	// controls.set({
	// 	// 	radius: 700,
	// 	// 	yaw: -0.2,
	// 	// 	pitch: 0.8,
	// 	// });

	// 	controls.zoomTo(pointcloud, {zoom: 5.0});
		
	// 	camera.near = 1;
	// 	camera.far = 10_000;
	// 	camera.updateProj();
	
	// 	window.pointcloud = pointcloud;

	// 	setPointcloud(pointcloud);
	// });

	// Potree.load("./resources/pointclouds/ca13/metadata.json").then(pointcloud => {

	// 	// controls.zoomTo(pointcloud);
	// 	controls.set({
	// 		yaw: -1.1,
	// 		pitch: 0.37,
	// 		radius: 406,
	// 		pivot: [696743.7622505882, 3919073.5328196282, 37.6882116012673],
	// 	});
		
	// 	camera.near = 1;
	// 	camera.far = 10_000;
	// 	camera.updateProj();
	
	// 	window.pointcloud = pointcloud;

	// 	setPointcloud(pointcloud);
	// });



	{
		let light1 = new PointLight("pointlight");
		light1.position.set(15, 15, 1);

		let light2 = new PointLight("pointlight2");
		light2.position.set(-15, -15, 1);

		scene.root.children.push(light1);
		// scene.root.children.push(light2);
	}

	// loadGLB("./resources/models/anita_mui.glb").then(node => {
	// // loadGLB("./resources/models/lion.glb").then(node => {
	// 	scene.root.children.push(node);

	// 	node.rotation.rotate(0.9 * Math.PI / 2, new Vector3(0, 1, 0));
	// 	node.position.set(5, 0, 3);
	// 	node.updateWorld();

	// 	controls.set({
	// 		yaw: -9.2375,
	// 		pitch: 0.2911333847340012,
	// 		radius: 3.649930853878021,
	// 		pivot:  [0.3169157776176301, -0.055293688684424885, 2.2],
	// 	});

	// 	window.glb = node;

	// 	// controls.zoomTo(node);
	// });

	requestAnimationFrame(loop);

}

run();
