import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { Heart, Activity } from "lucide-react";

interface FormData {
  gender: string;
  age: string;
  hypertension: string;
  heart_disease: string;
  ever_married: string;
  work_type: string;
  Residence_type: string;
  avg_glucose_level: string;
  bmi: string;
  smoking_status: string;
}

const StrokePredictionForm = () => {
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    gender: "",
    age: "",
    hypertension: "",
    heart_disease: "",
    ever_married: "",
    work_type: "",
    Residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
  });

  const handleInputChange = (field: keyof FormData, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:5001";
      const response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          age: parseInt(formData.age),
          avg_glucose_level: parseFloat(formData.avg_glucose_level),
          bmi: parseFloat(formData.bmi),
          hypertension: formData.hypertension === "Yes" ? 1 : 0,
          heart_disease: formData.heart_disease === "Yes" ? 1 : 0,
          ever_married: formData.ever_married === "Yes" ? 1 : 0,
        }),
      });

      const result = await response.json();

      if (!response.ok || result.error) {
        toast({
          title: "Error",
          description: result.error || "Failed to get prediction. Please check your input.",
          variant: "destructive",
        });
        return;
      }

      // result.data.prediction: 1 = High Risk, 0 = Low Risk
      // result.data.probability: { no_stroke, stroke }
      toast({
        title: "Prediction Result",
        description: `Predicted Stroke Risk: ${result.data.prediction === 1 ? "High Risk" : "Low Risk"}\nProbability of Stroke: ${(result.data.probability.stroke * 100).toFixed(2)}%\nProbability of No Stroke: ${(result.data.probability.no_stroke * 100).toFixed(2)}%`,
        variant: result.data.prediction === 1 ? "destructive" : "default",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to get prediction. Please check if the server is running.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = Object.values(formData).every(value => value !== "");

  return (
    <div className="min-h-screen bg-gradient-background flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl bg-gradient-card border-border/50 shadow-medical backdrop-blur-sm">
        <CardHeader className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-2">
            <Heart className="h-8 w-8 text-primary" />
            <Activity className="h-8 w-8 text-accent" />
          </div>
          <CardTitle className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            Stroke Risk Prediction
          </CardTitle>
          <CardDescription className="text-muted-foreground">
            Enter your health information to assess stroke risk using advanced machine learning
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Gender */}
              <div className="space-y-2">
                <Label htmlFor="gender">Gender</Label>
                <Select onValueChange={(value) => handleInputChange("gender", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select gender" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Male">Male</SelectItem>
                    <SelectItem value="Female">Female</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Age */}
              <div className="space-y-2">
                <Label htmlFor="age">Age</Label>
                <Input
                  id="age"
                  type="number"
                  min="1"
                  max="120"
                  value={formData.age}
                  onChange={(e) => handleInputChange("age", e.target.value)}
                  placeholder="Enter age"
                />
              </div>

              {/* Hypertension */}
              <div className="space-y-2">
                <Label htmlFor="hypertension">Hypertension</Label>
                <Select onValueChange={(value) => handleInputChange("hypertension", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Do you have hypertension?" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Heart Disease */}
              <div className="space-y-2">
                <Label htmlFor="heart_disease">Heart Disease</Label>
                <Select onValueChange={(value) => handleInputChange("heart_disease", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Do you have heart disease?" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Ever Married */}
              <div className="space-y-2">
                <Label htmlFor="ever_married">Ever Married</Label>
                <Select onValueChange={(value) => handleInputChange("ever_married", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Have you ever been married?" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Yes">Yes</SelectItem>
                    <SelectItem value="No">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Work Type */}
              <div className="space-y-2">
                <Label htmlFor="work_type">Work Type</Label>
                <Select onValueChange={(value) => handleInputChange("work_type", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select work type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Private">Private</SelectItem>
                    <SelectItem value="Self-employed">Self-employed</SelectItem>
                    <SelectItem value="Govt_job">Government Job</SelectItem>
                    <SelectItem value="Children">Children</SelectItem>
                    <SelectItem value="Never_worked">Never Worked</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Residence Type */}
              <div className="space-y-2">
                <Label htmlFor="Residence_type">Residence Type</Label>
                <Select onValueChange={(value) => handleInputChange("Residence_type", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select residence type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Urban">Urban</SelectItem>
                    <SelectItem value="Rural">Rural</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Average Glucose Level */}
              <div className="space-y-2">
                <Label htmlFor="avg_glucose_level">Average Glucose Level (mg/dL)</Label>
                <Input
                  id="avg_glucose_level"
                  type="number"
                  step="0.01"
                  min="0"
                  value={formData.avg_glucose_level}
                  onChange={(e) => handleInputChange("avg_glucose_level", e.target.value)}
                  placeholder="Enter glucose level"
                />
              </div>

              {/* BMI */}
              <div className="space-y-2">
                <Label htmlFor="bmi">BMI</Label>
                <Input
                  id="bmi"
                  type="number"
                  step="0.1"
                  min="0"
                  value={formData.bmi}
                  onChange={(e) => handleInputChange("bmi", e.target.value)}
                  placeholder="Enter BMI"
                />
              </div>

              {/* Smoking Status */}
              <div className="space-y-2 md:col-span-2">
                <Label htmlFor="smoking_status">Smoking Status</Label>
                <Select onValueChange={(value) => handleInputChange("smoking_status", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select smoking status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="formerly smoked">Formerly Smoked</SelectItem>
                    <SelectItem value="never smoked">Never Smoked</SelectItem>
                    <SelectItem value="smokes">Smokes</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Button 
              type="submit" 
              className="w-full bg-gradient-primary hover:shadow-glow transition-all duration-300 text-lg py-6"
              disabled={!isFormValid || isLoading}
            >
              {isLoading ? "Analyzing..." : "Predict Stroke Risk"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default StrokePredictionForm;