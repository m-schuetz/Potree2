
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {render} from "./potree/renderQuads.js";
import {EventDispatcher} from "./misc/EventDispatcher.js";
import {Vector3} from "./math/math.js";

async function load(url, args = {}){
	let octree = await PotreeLoader.load(url);

	octree.name = args.name ?? "octree";

	return octree;
}

const pickPosition = new Vector3();
const pickQueue = [];

function pick(x, y, callback){
	pickQueue.push({x, y, callback});
}

export * from "./math/math.js";
export * from "./core/Geometry.js";
export * from "./core/RenderTarget.js";
export * from "./scene/Scene.js";
export * from "./scene/SceneNode.js";
export * from "./scene/PointLight.js";
export * from "./modules/mesh/Mesh.js";
export * from "./modules/mesh/NormalMaterial.js";
export * from "./modules/mesh/PhongMaterial.js";
export * from "./modules/mesh/WireframeMaterial.js";
export * from "./modules/mesh/TriangleColorMaterial.js";
export * from "./scene/Camera.js";
export * from "./navigation/OrbitControls.js";
export * from "./navigation/PotreeControls.js";
export * from "./renderer/Renderer.js";
export * from "./modules/points/Points.js";
export * as Timer from "./renderer/Timer.js";
export * from "./potree/PointCloudOctree.js";
export * from "./potree/PointCloudOctreeNode.js";
export * from "./modules/mesh/renderMesh.js";
export * from "./modules/quads/renderQuads.js";
export {load as loadGLB} from "./misc/GLBLoader.js";
export * from "./utils.js";
export * from "./defines.js";
export * from "./misc/Gradients.js";
export * from "./misc/WorkerPool.js";
export * from "./misc/EventDispatcher.js";
export * from "./InputHandler.js";
export * from "./potree/CopcLoader.js";
export * from "./potree/images/Images360.js";

export {render as renderPoints} from "./prototyping/renderPoints.js";
export {render as renderPointsOctree}  from "./potree/renderPointsOctree.js";
export {render as renderQuadsOctree}  from "./potree/renderQuadsOctree.js";
export {dilate}  from "./potree/dilate.js";
export {EDL}  from "./potree/EDL.js";
export {hqs_normalize}  from "./potree/hqs_normalize.js";

import {createPointsData, createPointsSphere} from "./modules/geometries/points.js";
import {cube} from "./modules/geometries/cube.js";
import {sphere} from "./modules/geometries/sphere.js";

export const geometries = {createPointsData, createPointsSphere, cube, sphere};
export * as math from "./math/math.js";

import {init} from "./init.js";
import {load as loadGLB} from "./misc/GLBLoader.js";
import * as Gradients from "./misc/Gradients.js";

const dispatcher = new EventDispatcher();

const events = {
	dispatcher,
	onPointcloudLoaded: (callback, args) => dispatcher.add("pointcloud_loaded", callback, args),
	onRootNodeLoaded: (callback, args) => dispatcher.add("root_node_loaded", callback, args),
};

const settings = {
	pointSize: 3,
	pointBudget: 2_000_000,
	attribute: "rgba",
	dbgAttribute: "rgba",
	debugU: 1,
	showBoundingBox: false,
	useCompute: false,
	dilateEnabled: false,
	edlEnabled: false,
	updateEnabled: true,
	gradient: Gradients.SPECTRAL,
};

const state = {
	fps: 0,
	camPos: "",
	camDir: "",
	numPoints: 0,
	numVoxels: 0,
	numNodes: 0,
	renderedObjects: [],
	renderedElements: 0,
};

export let Potree = {
	load, loadGLB,
	render: render,
	pick, pickQueue, pickPosition,
	init,
	settings, state,
	events,
	scene: null,

	version: "2.0 (WebGPU prototype)",
};


