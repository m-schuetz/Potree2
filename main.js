

import {Renderer} from "./src/renderer/Renderer.js";
import {Camera} from "./src/scene/Camera.js";
import {mat4, vec3} from './libs/gl-matrix.js';
import {OrbitControls} from "./src/navigation/OrbitControls.js";
import {Lines} from "./src/modules/lines/Lines.js";
import {render as renderLines} from "./src/modules/lines/render.js";
import {Geometry} from "./src/core/Geometry.js";
import {Vector3} from "./src/math/Vector3.js";

import {Potree} from "./src/Potree.js";

import {render as renderQuads}  from "./src/potree/renderQuads.js";
import {render as renderPoints}  from "./src/potree/renderPoints.js";

import * as dat from "./libs/dat.gui/dat.gui.module.js";

let frame = 0;
let lastFrameStart = 0;
let lastFpsCount = 0;
let framesSinceLastCount = 0;
let fps = 0;

let primitiveType = "points";

let gui = null;
let guiContent = {
	"show bounding box": false,
	"primitive": "points",
	"#points": "0",
	"#nodes": "0",
	"fps": "0",
};


function initGUI(){

	gui = new dat.GUI();
	
	{
		let stats = gui.addFolder("stats");
		stats.open();
		stats.add(guiContent, "#points").listen();
		stats.add(guiContent, "#nodes").listen();
		stats.add(guiContent, "fps").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		input.add(guiContent, "primitive", ["points", "quads"]);
		input.add(guiContent, "show bounding box");

		// slider
		//input.add(text, 'fontSize', 6, 48).onChange(setValue);
	}

}

async function run(){

	initGUI();

	let renderer = new Renderer();

	window.renderer = renderer;

	await renderer.init();

	let camera = new Camera();

	let controls = new OrbitControls(renderer.canvas);
	// controls.radius = 30;
	// controls.yaw = Math.PI / 4;
	// controls.pitch = Math.PI / 5;

	Potree.load("./resources/pointclouds/lion/metadata.json").then(pointcloud => {

		controls.radius = 10;
		controls.yaw = -Math.PI / 6;
		controls.pitch = Math.PI / 5;

		pointcloud.updateVisibility(camera);
		pointcloud.position.set(-0.9, 0.1, -5);
		pointcloud.updateWorld();
		window.pointcloud = pointcloud;
	});

	// Potree.load("./resources/pointclouds/heidentor/metadata.json").then(pointcloud => {
	// 	controls.radius = 30;
	// 	controls.yaw = Math.PI / 4;
	// 	controls.pitch = Math.PI / 5;
	//
	// 	pointcloud.updateVisibility(camera);
	// 	pointcloud.position.set(3, -3, -6)
	// 	pointcloud.updateWorld();
	// 	window.pointcloud = pointcloud;
	// });

	let lines = null;
	{
		let geometry = new Geometry();
		geometry.buffers = [{
			name: "position",
			buffer: new Float32Array([
				-1, -1, -1,
				1, 1, 1,
				0, 0, 0,
				6, 3, 1
			]),
		}];
		geometry.numElements = 4;

		lines = new Lines("test", geometry);

	}
	

	let loop = () => {
		// let timeSinceLastFrame = performance.now() - lastFrameStart;
		// console.log(timeSinceLastFrame.toFixed(1));
		// lastFrameStart = performance.now();
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
		mat4.copy(camera.world, controls.world);
		camera.updateView();

		let size = renderer.getSize();
		camera.aspect = size.width / size.height;
		camera.updateProj();

		{ // draw to window
			let pass = renderer.start();

			if(window.pointcloud){
				let pointcloud = window.pointcloud;

				pointcloud.updateVisibility(camera);
				pointcloud.showBoundingBox = guiContent["show bounding box"];

				let numPoints = pointcloud.visibleNodes.map(n => n.geometry.numElements).reduce( (a, i) => a + i, 0);
				let numNodes = pointcloud.visibleNodes.length;

				guiContent["#points"] = numPoints.toLocaleString();
				guiContent["#nodes"] = numNodes.toLocaleString();

				//Potree.render(renderer, pass, window.pointcloud, camera);

				if(guiContent.primitive === "points"){
					renderPoints(renderer, pass, pointcloud, camera);
				}else if(guiContent.primitive === "quads"){
					renderQuads(renderer, pass, pointcloud, camera);
				}
			}

			// if(lines){
			// 		renderLines(renderer, pass, lines, camera);
			// }

			// renderer.drawBoundingBox(
			// 	new Vector3(1, 2, 3), 
			// 	new Vector3(0.3, 0.3, 0.3), 
			// 	new Vector3(1, 0, 1)
			// );
			
			

			renderer.renderDrawCommands(pass, camera);
			renderer.finish(pass);
		}

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);

}


run();





