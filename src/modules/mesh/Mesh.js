
import {SceneNode} from "../../scene/SceneNode.js";
import {NormalMaterial} from "./NormalMaterial.js";

export class Mesh extends SceneNode{

	constructor(name, geometry){
		super(name);

		this.geometry = geometry;
		this.material = new NormalMaterial();
	}

}

