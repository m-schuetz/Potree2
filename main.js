

import {cube, pointCube} from "./src/prototyping/cube.js";

import {Renderer} from "./src/renderer/Renderer.js";
import {loadImage, drawImage} from "./src/prototyping/textures.js";
import {drawRect} from "./src/prototyping/rect.js";
import {drawMesh} from "./src/modules/mesh/drawMesh.js";
import {Geometry} from "./src/core/Geometry.js";
import {Mesh} from "./src/modules/mesh/Mesh.js";
import {Points} from "./src/modules/points/Points.js";
import {drawPoints} from "./src/modules/points/drawPoints.js";
import {drawQuads} from "./src/modules/points/drawQuads.js";
import {Camera} from "./src/scene/Camera.js";
import {mat4, vec3} from '../libs/gl-matrix.js';
import {OrbitControls} from "./src/navigation/OrbitControls.js";

import {SceneNode} from "./src/scene/SceneNode.js";

let frame = 0;

async function run(){

	let renderer = new Renderer();

	window.renderer = renderer;

	await renderer.init();

	let camera = new Camera();
	let cameraDistance = 5;
	mat4.translate(camera.world, camera.world, vec3.fromValues(0, 0, cameraDistance));
	camera.updateView();

	let controls = new OrbitControls(renderer.canvas);
	controls.radius = 4;
	controls.yaw = Math.PI / 4;
	controls.pitch = Math.PI / 5;

	let mesh = new Mesh("test", cube.buffers, cube.vertexCount);
	mesh.scale.set(0.3, 0.3, 0.3);
	mesh.position.set(0, 0, 0);
	mesh.updateWorld();

	let geometry = new Geometry();
	geometry.buffers = pointCube.buffers;
	geometry.numElements = pointCube.vertexCount;
	let points = new Points("test", geometry);
	// points.scale.set(0.5, 0.5, 0.5);
	// points.position.set(2, 0, 0);
	points.updateWorld();

	let image = await loadImage("./resources/images/background.jpg");

	let loop = () => {
		frame++;

		controls.update();
		mat4.copy(camera.world, controls.world);
		camera.updateView();

		let size = renderer.getSize();
		camera.aspect = size.width / size.height;
		camera.updateProj();

		{
			let pass = renderer.start();

			drawQuads(renderer, pass, points, camera);
			drawMesh(renderer, pass, mesh, camera);
			// drawPoints(renderer, pass, points, camera);
			// drawRect(renderer, pass, -0.8, -0.8, 0.2, 0.5);
			// drawImage(renderer, pass, image, 0.3, 0.3, 0.4, -0.4);

			renderer.finish(pass);
		}

		requestAnimationFrame(loop);
	};
	requestAnimationFrame(loop);

}


run();





