
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {render} from "./potree/renderQuads.js";


async function load(url, args = {}){
	let octree = await PotreeLoader.load(url);

	octree.name = args.name ?? "octree";

	return octree;
}


const pickQueue = [];

function pick(x, y, callback){
	pickQueue.push({x, y, callback});
}

export * from "./math/math.js";
export * from "./core/Geometry.js";
export * from "./core/RenderTarget.js";
export * from "./scene/Scene.js";
export * from "./scene/SceneNode.js";
export * from "./modules/mesh/Mesh.js";
export * from "./scene/Camera.js";
export * from "./navigation/OrbitControls.js";
export * from "./renderer/Renderer.js";
export * from "./modules/points/Points.js";
export * as Timer from "./renderer/Timer.js";
export * from "./potree/PointCloudOctree.js";
export * from "./potree/PointCloudOctreeNode.js";
export * from "./modules/mesh/renderMesh.js";
export {load as loadGLB} from "./misc/GLBLoader.js";
export * from "./misc/Gradients.js";

export * from "./misc/EventDispatcher.js";

export {render as renderPoints} from "./prototyping/renderPoints.js";
export {render as renderPointsCompute} from "./prototyping/renderPointsCompute.js";
export {render as renderPointsOctree}  from "./potree/renderPointsOctree.js";
export {render as renderPointsOctreeBundledVBO}  from "./potree/renderPointsOctreeBundledVBO.js";
export {dilate}  from "./potree/dilate.js";
export {EDL}  from "./potree/EDL.js";
export {hqs_normalize}  from "./potree/hqs_normalize.js";


import {createPointsData, createPointsSphere} from "./modules/geometries/points.js";
import {cube} from "./modules/geometries/cube.js";
export const geometries = {createPointsData, createPointsSphere, cube};

import {init} from "./init.js";

import {load as loadGLB} from "./misc/GLBLoader.js";

const settings = {
	pointSize: 3,
	pointBudget: 1_000_000,
	attribute: "rgba",
	showBoundingBox: false,
	// mode: "pixels",
	useCompute: false,
	dilateEnabled: false,
	edlEnabled: false,
};

const state = {
	fps: 0,
	camPos: "",
	camDir: "",
	numPoints: 0,
	numNodes: 0,
};

export let Potree = {
	load: load,
	loadGLB: loadGLB,
	render: render,
	pick: pick, pickQueue,
	init: init,
	settings: settings,
	state: state,
};



