
import {SceneNode} from "../../scene/SceneNode.js";

export class Quads extends SceneNode{

	constructor(name, positions){
		super(name);

		this.positions = positions;
		this.size = 20;

	}

}

