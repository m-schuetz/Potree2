
import {Gradients, Utils} from "potree";
let dir = new URL(import.meta.url + "/../").href;

class Panel{

	constructor(){
		this.element = document.createElement("div");
		this.elTable = document.createElement("div");

		let elTitle = document.createElement("div");
		elTitle.classList.add("subsection");
		elTitle.textContent = "Scene";

		this.element.append(elTitle);
		this.element.append(this.elTable);

		this.oldTableString = "";
	}

	set(scene){
		this.scene = scene;
	}

	update(){

		let scene = Potree.instance.scene;

		let tableString = "";

		let nodes = [];

		scene.root.traverse(node => {

			let i = nodes.length;
			
			if(node.constructor.name === "PointCloudOctree"){
				let label = node.name ?? "&lt;unnamed pointcloud&gt;";
				tableString += `
					<span> </span>
					<span>${label}</span>
					<span name="zoom_${i}" style="cursor: pointer;">◉</span>
					<span><input type="checkbox" name="item_${i}"></span>
				`;
				// <span><input type="button" class="textButton" name="zoom_${i}" value="◉"/></span>

				nodes.push(node);
			}else if(node.constructor.name === "Images360"){
				let label = node.name ?? "&lt;360° Images&gt;";
				tableString += `
					<span> </span>
					<span>${label}</span>
					<span name="zoom_${i}" style="cursor: pointer;">◉</span>
					<span><input type="checkbox" name="item_${i}"></span>
				`;

				nodes.push(node);
			}else if(node.constructor.name === "TDTiles"){
				let label = node.name ?? "&lt;3D Tiles&gt;";
				tableString += `
					<span> </span>
					<span>${label}</span>
					<span name="zoom_${i}" style="cursor: pointer;">◉</span>
					<span><input type="checkbox" name="item_${i}"></span>
				`;

				nodes.push(node);
			}

		});

		if(tableString !== this.oldTableString){
			this.elTable.style.display = "grid";
			this.elTable.style.gridTemplateColumns = "1em 1fr 1em 1em";
			this.elTable.innerHTML = tableString;

			this.oldTableString = tableString;

			for(let i = 0; i < nodes.length; i++){
				let node = nodes[i];
				let elCheckbox = this.elTable.querySelector(`input[name=item_${i}]`);
				let elZoom = this.elTable.querySelector(`input[name=zoom_${i}]`)
					?? this.elTable.querySelector(`span[name=zoom_${i}]`)
				
				elCheckbox.checked = node.visible;

				elCheckbox.onclick = () => {
					node.visible = elCheckbox.checked;
				};

				if(elZoom){
					elZoom.onclick = () => {
						potree.controls.zoomTo(node);
					};
				}


			}
		}

		requestAnimationFrame(this.update.bind(this));
	}

}

export function createPanel(){
	let panel = new Panel();
	panel.element.id = "scene_panel";
	panel.element.classList.add("subsection_panel");

	panel.update();

	return panel;
}