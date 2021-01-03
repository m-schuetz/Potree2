
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {PointCloudOctree} from "./potree/PointCloudOctree.js";



async function load(url){
	let octree = await PotreeLoader.load(url);

	return octree;
}



export let Potree = {
	load: load,
};