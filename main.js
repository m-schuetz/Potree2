

import {cube, pointCube} from "./src/prototyping/cube.js";

import {Renderer} from "./src/renderer/Renderer.js";
import {loadImage, drawImage} from "./src/prototyping/textures.js";
import {drawRect} from "./src/prototyping/rect.js";
import {drawMesh} from "./src/prototyping/drawMesh.js";
import {drawPoints} from "./src/prototyping/drawPoints.js";
import {drawQuads} from "./src/prototyping/drawQuads.js";
import {Camera} from "./src/scene/Camera.js";
import {mat4, vec3} from '../libs/gl-matrix.js';
import {OrbitControls} from "./src/navigation/OrbitControls.js";

import {SceneNode} from "./src/scene/SceneNode.js";

let frame = 0;

let testNode = new SceneNode("test");
{

	testNode.scale.set(0.1, 0.1, 0.1);
	testNode.position.set(0, 1, 0);
	testNode.updateWorld();
}


async function run(){

	let renderer = new Renderer();

	window.renderer = renderer;

	await renderer.init();

	let node = pointCube;
	let camera = new Camera();
	let cameraDistance = 5;
	mat4.translate(camera.world, camera.world, vec3.fromValues(0, 0, cameraDistance));
	camera.updateView();

	let controls = new OrbitControls(renderer.canvas);
	controls.radius = 4;
	controls.yaw = Math.PI / 4;
	controls.pitch = Math.PI / 5;

	let image = await loadImage("./resources/images/background.jpg");

	let loop = () => {
		frame++;

		controls.update();
		mat4.copy(camera.world, controls.world);
		camera.updateView();

		let pass = renderer.start();

			// drawPoints(renderer, pass, node, camera);
			drawQuads(renderer, pass, node, camera);
			drawMesh(renderer, pass, cube, camera, testNode.world);
			drawRect(renderer, pass, -0.8, -0.8, 0.2, 0.5);
			drawImage(renderer, pass, image, 0.3, 0.3, 0.4, -0.4);

		renderer.finish(pass);

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);

}


run();





