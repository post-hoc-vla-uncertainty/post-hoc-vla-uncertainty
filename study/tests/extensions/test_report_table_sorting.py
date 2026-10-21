"""Column clicks work without loading the inspector; sort estimates, not CI text."""
from html.parser import HTMLParser
import json
import shutil
import subprocess

import pytest

from head_report.html import TABLE_JS, document, table


class Table(HTMLParser):
    def __init__(self):
        super().__init__();self.rows=[];self.row=None;self.sort_values=[];self.buttons=0
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='tr':self.row=[]
        if tag=='td':self.row.append(values['data-sort'])
        if tag=='button' and values.get('class')=='column-sort':self.buttons+=1
    def handle_endtag(self,tag):
        if tag=='tr' and self.row:self.rows.append(self.row)


def test_sort_buttons_load_before_inspector_and_expose_raw_numeric_means():
    body=table([dict(label='A',value=.12,value_sd=.9,value_total_low=-1,value_total_high=1)],
               [('label','Configuration'),('value','Mean ± SD')])
    parser=Table();parser.feed(body)
    assert parser.buttons==2 and parser.rows==[['A','0.12']]
    assert 'aria-sort="none"' in body and 'class="sort-indicator"' in body
    html=document(body,'invalid inspector JSON',[])
    assert html.index(TABLE_JS)<html.index('<script id="report-data"')


@pytest.mark.skipif(shutil.which('node') is None,reason='Node is needed to exercise the emitted JavaScript')
def test_numeric_text_toggle_missing_values_and_filter_without_inspector():
    body=table([dict(label='zebra',value=.12,value_sd=100),dict(label='alpha',value=.9,value_sd=0),
                dict(label='middle',value=-.2,value_sd=1),dict(label='missing',value=None),dict(label='ten',value=10)],
               [('label','Head'),('value','Mean ± SD')])
    parsed=Table();parsed.feed(body)
    script=r'''
const vm=require('vm'),assert=require('assert'),fs=require('fs');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
class Element {
 constructor(){this.handlers={};this.attributes={};this.textContent='';this.hidden=false;}
 addEventListener(name,callback){this.handlers[name]=callback;}
 setAttribute(name,value){this.attributes[name]=value;}
 getAttribute(name){return this.attributes[name]||null;}
}
const rows=input.rows.map(values=>({cells:values.map(sort=>({dataset:{sort}})),textContent:values.join(' '),hidden:false}));
const body={rows:[...rows],appendChild(row){this.rows.splice(this.rows.indexOf(row),1);this.rows.push(row);}};
const headers=[0,1].map(column=>{const th=new Element();th.dataset={column:String(column)};th.button=new Element();th.indicator=new Element();th.setAttribute('aria-sort','none');th.querySelector=selector=>selector==='button.column-sort'?th.button:th.indicator;return th;});
const table={tBodies:[body],querySelectorAll:()=>headers};
const search=new Element();search.value='';const count=new Element(),note=new Element();
const document={querySelectorAll:selector=>selector==='table.sortable'?[table]:body.rows,getElementById:id=>{
 if(id==='report-data')throw Error('Table initialization must not read inspector data');
 return {'search':search,'visible-count':count,'table-interactivity-note':note}[id];
}};
vm.runInNewContext(input.script,{document});
const labels=()=>body.rows.map(r=>r.cells[0].dataset.sort);
headers[1].button.handlers.click();assert.deepStrictEqual(labels(),['middle','zebra','alpha','ten','missing']);
assert.strictEqual(headers[1].getAttribute('aria-sort'),'ascending');assert.strictEqual(headers[1].indicator.textContent,'↑');
headers[1].button.handlers.click();assert.deepStrictEqual(labels(),['ten','alpha','zebra','middle','missing']);
assert.strictEqual(headers[1].getAttribute('aria-sort'),'descending');assert.strictEqual(headers[1].indicator.textContent,'↓');
headers[0].button.handlers.click();assert.deepStrictEqual(labels(),['alpha','middle','missing','ten','zebra']);
assert.strictEqual(headers[1].getAttribute('aria-sort'),'none');assert.strictEqual(headers[1].indicator.textContent,'↕');
search.value='ALPHA';search.handlers.input();assert.deepStrictEqual(body.rows.filter(r=>!r.hidden).map(r=>r.cells[0].dataset.sort),['alpha']);
assert.strictEqual(count.textContent,'1 configuration groups shown');assert.strictEqual(note.hidden,true);
console.log('Column clicks, ascending/descending numeric means, missing cells, text sorting, and filtering passed.');
'''
    result=subprocess.run(['node','-e',script],input=json.dumps(dict(rows=parsed.rows,script=TABLE_JS)),
                          text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
