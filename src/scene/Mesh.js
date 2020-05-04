
import {SceneNode} from "./SceneNode.js";

export class Mesh extends SceneNode{

	constructor(name, geometry){
		super(name);

		this.geometry = geometry;
	}

}


