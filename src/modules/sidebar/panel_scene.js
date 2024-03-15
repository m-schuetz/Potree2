
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

		// let tableString = "";
		tableString += `<table style="border-collapse: collapse; "> \n`;
		tableString += `	<tr> \n`;
		tableString += `		<th></th> \n`;
		tableString += `		<th style="text-align: left;"></th> \n`;
		tableString += `		<th></th> \n`;
		tableString += `		<th></th> \n`;
		tableString += `	</tr> \n`;

		scene.root.traverse(node => {
			let i = nodes.length;

			let isValidSceneObject = [
				"PointCloudOctree",
				"Images360",
				"TDTiles"
			].includes(node.constructor.name);

			if(isValidSceneObject){
				tableString += `
					<tr>
						<td style="width: 10px; padding: 0;"></td>
						<td name="label_${i}" style="cursor: pointer; width: 100%"></td>
						<td name="zoom_${i}" style="cursor: pointer;">◉</td>
						<td><input type="checkbox" name="item_${i}"></td>
					</tr>
				`;
				nodes.push(node);
			}
		});

		if(tableString !== this.oldTableString){
			this.elTable.style.display = "grid";
			this.elTable.innerHTML = tableString;
			this.oldTableString = tableString;

			for(let i = 0; i < nodes.length; i++){
				let node = nodes[i];

				let elLabel    = this.elTable.querySelector(`td[name=label_${i}]`);
				let elCheckbox = this.elTable.querySelector(`input[name=item_${i}]`);
				let elZoom     = this.elTable.querySelector(`input[name=zoom_${i}]`)
					?? this.elTable.querySelector(`td[name=zoom_${i}]`);

				
				elLabel.innerHTML = node.name ?? `&lt;${node.constructor.name} object&gt;`;
				

				if(elLabel){
					elLabel.onmouseenter = () => {node.isHighlighted = true;};
					elLabel.onmouseleave = () => {node.isHighlighted = false;};
				}

				if(elZoom){
					elZoom.onmouseenter = () => {node.isHighlighted = true;};
					elZoom.onmouseleave = () => {node.isHighlighted = false;};
				}
				
				if(elCheckbox){
					elCheckbox.checked = node.visible;
					elCheckbox.onclick = () => {
						node.visible = elCheckbox.checked;
					};

					elCheckbox.onmouseenter = () => {node.isHighlighted = true;};
					elCheckbox.onmouseleave = () => {node.isHighlighted = false;};
				}

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