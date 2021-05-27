# flexassist-core
Repository for AI related technical workflow and demos

# Installation
Install Libraries and Dependencies 
```sh
# installs the library in 'edit' mode for development into prior installed environment
# show be installed as: `flexassist-core` in listing the packages (conda list |less )
cd flexassist-core
rm -r $(find . -name '*.egg-info')
pip uninstall <package_name> 
pip install -e .  
```

## Citations
------------------
If you reference in your research, please cite:
```
@article{FlexAssist2021,
    author = {Kamlani, Ari},
    title  = {{Spinning up core Machine Learning and Deep Learning Frameworks}},
    year   = {2021}
}
