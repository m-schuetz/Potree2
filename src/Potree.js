
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {PointCloudOctree} from "./potree/PointCloudOctree.js";
// import {render} from "./potree/renderPoints.js";
import {render} from "./potree/renderQuads.js";


async function load(url){
	let octree = await PotreeLoader.load(url);

	return octree;
}

export let Potree = {
	load: load,
	render: render,
};