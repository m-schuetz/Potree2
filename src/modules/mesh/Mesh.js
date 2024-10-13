
import {SceneNode} from "../../scene/SceneNode.js";
import {NormalMaterial} from "./NormalMaterial.js";
import {renderMeshes} from "potree";

export class Mesh extends SceneNode{

	constructor(name, geometry){
		super(name);

		this.geometry = geometry;
		this.material = new NormalMaterial();

		if(geometry.boundingBox){
			this.boundingBox.copy(geometry.boundingBox);
		}
	}

	render(drawstate){
		renderMeshes([this], drawstate);
	}

}

